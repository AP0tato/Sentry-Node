from ultralytics import YOLO
import time
import queue
import threading
import subprocess
import sys
import os
import glob
import re
import shutil
import torch
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import atexit

# ── Device detection ──────────────────────────────────────────────────────────
device = "cpu"
batch  = 8
epochs = 50
imgsz  = 1280

if torch.cuda.is_available():
    batch, epochs, imgsz, device = 4, 100, 1280, "cuda"
elif torch.backends.mps.is_available():
    batch, epochs, imgsz, device = 8, 100, 1280, "mps"

# ─────────────────────────────────────────────────────────────────────────────
#  THREAD-SAFE LOG QUEUE
# ─────────────────────────────────────────────────────────────────────────────
ANSI_RE  = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
log_queue: "queue.Queue[str]" = queue.Queue()

class QueueWriter:
    def __init__(self, original):
        self.original = original
        self._lock    = threading.Lock()
    def write(self, text):
        if not text: return
        try:
            self.original.write(text)
            self.original.flush()
        except: pass
        # Do not strip ANSI here. Keep GUI stripping centralized in _poll_log.
        with self._lock: log_queue.put(text)
    def flush(self):
        try: self.original.flush()
        except: pass
    def fileno(self): return self.original.fileno()

# ── Training worker ───────────────────────────────────────────────────────────
current_process = None
stop_flag = False

def _run_training(yaml_path, model_path, ep, img, bat, dev, workers, lr0, lrf, warmup, resume, app):
    global current_process, stop_flag
    stop_flag = False
    try:
        app.set_status("Training…", "yellow")
        yolo_exe = os.path.join(os.path.dirname(sys.executable), "yolo.exe")
        yolo_cmd = yolo_exe if os.path.exists(yolo_exe) else (shutil.which("yolo") or "yolo")

        cmd = [
            yolo_cmd, "train", f"data={yaml_path}", f"model={model_path}",
            f"epochs={ep}", f"imgsz={img}", f"batch={bat}", f"device={dev}",
            f"workers={workers}", f"lr0={lr0}", f"lrf={lrf}", f"warmup_epochs={warmup}",
            "name=microplastics_run"
        ]
        if resume: cmd.append("resume=True")

        current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        )

        if current_process.stdout:
            # Reconfigure the real stdout to UTF-8 so Unicode chars from the
            # subprocess (e.g. box-drawing, emoji) don't hit a charmap error.
            try:
                sys.__stdout__.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
            with open("subprocess_chunks.log", "w", encoding="utf-8") as f:
                # readline() only returns on \n, so YOLO's \r-terminated
                # progress updates get held in the buffer until the next \n,
                # causing them to arrive in a batch and print on new lines.
                # Instead we accumulate chars and flush the segment to the
                # queue on every \r or \n so the GUI sees each update promptly.
                buf = []
                while True:
                    ch = current_process.stdout.read(1)
                    if ch == "":
                        # EOF — flush whatever remains
                        if buf:
                            segment = "".join(buf)
                            try:
                                sys.__stdout__.write(segment)
                                sys.__stdout__.flush()
                            except Exception:
                                pass
                            f.write(f"SEG: {repr(segment)}\n")
                            f.flush()
                            log_queue.put(segment)
                        break
                    buf.append(ch)
                    if ch in ("\r", "\n"):
                        segment = "".join(buf)
                        buf = []
                        try:
                            sys.__stdout__.write(segment)
                            sys.__stdout__.flush()
                        except Exception:
                            pass
                        f.write(f"SEG: {repr(segment)}\n")
                        f.flush()
                        log_queue.put(segment)

        return_code = current_process.wait()
        
        if stop_flag:
            app.set_status("Stopped", "orange")
        elif return_code == 0:
            app.set_status("Done ✓", "lime green")
            # Post-train Inference
            test_img = os.path.join(os.getcwd(), "test_image.jpg")
            if os.path.exists(test_img):
                weight_candidates = ["runs/detect/microplastics_run/weights/best.pt", "runs/detect/microplastics_run/weights/last.pt"]
                weight_path = next((c for c in weight_candidates if os.path.exists(c)), None)
                
                if weight_path:
                    print(f"\n[INFO] Running test on {test_img}\n")
                    model = YOLO(weight_path)
                    res = model.predict(test_img, verbose=False)
                    if res:
                        cv2.imshow("Result", res[0].plot())
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
        else:
            app.set_status("Error — see log", "red")
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        app.set_status("Error", "red")
    finally:
        current_process = None
        app.training_finished()

# ── Test image inference ─────────────────────────────────────────────────────
def _run_test_image(app, conf=0.25):
    """Finds best.pt and runs inference on test_image.jpg, shows a large result window."""
    try:
        app.set_status("Running inference…", "yellow")

        test_img = os.path.join(os.getcwd(), "test_image.jpg")
        if not os.path.exists(test_img):
            print("[ERROR] test_image.jpg not found in current directory.\n")
            app.set_status("Idle", "#888")
            return

        # find the most recently modified best.pt
        checkpoints = glob.glob("runs/detect/*/weights/best.pt")
        if not checkpoints:
            print("[ERROR] No best.pt found — train first.\n")
            app.set_status("Idle", "#888")
            return

        weight_path = max(checkpoints, key=os.path.getmtime)
        print(f"[INFO] Model  : {weight_path}")
        print(f"[INFO] Image  : {test_img}")
        print(f"[INFO] Conf   : {conf}\n")

        model   = YOLO(weight_path)
        results = model.predict(test_img, conf=conf, verbose=True)

        if not results:
            print("[ERROR] No results returned.\n")
            app.set_status("Idle", "#888")
            return

        res       = results[0]
        annotated = res.plot()

        # print detection summary
        from collections import Counter
        names  = [res.names[int(b.cls)] for b in res.boxes]
        counts = Counter(names)
        print(f"\n─── Detection Summary ───────────────────")
        print(f"  Total detected : {len(names)}")
        for cls, n in counts.most_common():
            pct = n / len(names) * 100 if names else 0
            print(f"  {cls:<25} {n:>3}  ({pct:.1f}%)")
        print("─────────────────────────────────────────\n")

        # scale to a large window (max 1200x900) while keeping aspect ratio
        h, w   = annotated.shape[:2]
        scale  = min(1200 / w, 900 / h, 1.0)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        display = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Test Image — Detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Test Image — Detections", new_w, new_h)
        cv2.imshow("Test Image — Detections", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        app.set_status("Done ✓", "lime green")

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}\n")
        app.set_status("Error — see log", "red")


# ── Main GUI ──────────────────────────────────────────────────────────────────
class TrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Microplastics YOLO Trainer")
        self.configure(bg="#1a1a2e")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._build_styles()
        self._build_ui()

        sys.stdout = QueueWriter(sys.__stdout__)
        sys.stderr = QueueWriter(sys.__stderr__)

        self._log_partial, self._ansi_in_progress, self._pending_r = "", False, False
        self._line_open = False  # True when a \x1b[K line is held open (no \n)
        # Track the start index of the current last (partial) line so we can
        # delete/overwrite it when a '\r' update arrives.
        self._last_line_start = None
        self._poll_log()
        atexit.register(self.cleanup_subprocess)

    def on_closing(self):
        self.cleanup_subprocess()
        self.destroy()
        sys.exit(0)

    def cleanup_subprocess(self):
        global current_process
        if current_process and current_process.poll() is None:
            try:
                current_process.terminate()
                current_process.wait(timeout=2)
            except:
                try: current_process.kill()
                except: pass
            current_process = None

    def _safe_linestart(self):
        try:
            return self.log_box.index("end-1c linestart")
        except Exception:
            return None

    def _poll_log(self):
        parts = []
        while not log_queue.empty():
            parts.append(log_queue.get_nowait())

        if parts:
            self.log_box.configure(state="normal")
            for chunk in parts:
                # YOLO uses \x1b[K (erase-to-end-of-line) before each progress
                # update instead of \r.  Detect it BEFORE stripping ANSI so we
                # know to overwrite the last line rather than append a new one.
                erase_line = '\x1b[K' in chunk

                # Strip all ANSI escape sequences
                cleaned = ANSI_RE.sub('', chunk)

                # Normalize \r\n → \n, handle backspaces
                cleaned = cleaned.replace('\r\n', '\n')
                buf = []
                for ch in cleaned:
                    if ch == '\x08':
                        if buf: buf.pop()
                    else:
                        buf.append(ch)
                cleaned = ''.join(buf)

                # A bare \r also means overwrite
                overwrite = erase_line or cleaned.startswith('\r')
                cleaned = cleaned.lstrip('\r')

                if not cleaned:
                    continue

                # If this chunk is NOT an overwrite but the previous \x1b[K line
                # was left open (no \n emitted), close it now so this content
                # starts on a fresh line instead of being appended mid-line.
                if not overwrite and self._line_open:
                    self.log_box.insert(tk.END, '\n')
                    self._last_line_start = None
                    self._line_open = False

                # Split on \n keeping the delimiter tokens
                parts_inner = re.split(r'(\n)', cleaned)
                first = True
                j = 0
                while j < len(parts_inner):
                    token = parts_inner[j]

                    if token == '\n':
                        self.log_box.insert(tk.END, '\n')
                        self._last_line_start = None
                        self._line_open = False
                        j += 1
                        first = False
                        continue

                    if not token:
                        j += 1
                        continue

                    if overwrite and first:
                        # Delete the previous progress line and write new content
                        try:
                            start = self._last_line_start or self._safe_linestart()
                            self.log_box.delete(start, "end-1c")
                        except Exception:
                            start = self._safe_linestart()
                        self.log_box.insert(tk.END, token)
                        if self._last_line_start is None:
                            self._last_line_start = self._safe_linestart()
                    else:
                        if self._last_line_start is None:
                            self._last_line_start = self._safe_linestart()
                        self.log_box.insert(tk.END, token)

                    first = False
                    j += 1

                # If this was an overwrite chunk, its \n was the last token and
                # was already emitted above.  But YOLO's progress lines end with
                # \n which we DO want to suppress so the next \x1b[K overwrites
                # the same line.  Re-check: if erase_line and the cleaned text
                # ended with \n, that \n was already written — undo it by
                # deleting the last char and marking the line as open.
                if erase_line and cleaned.endswith('\n'):
                    try:
                        self.log_box.delete("end-2c", "end-1c")  # remove the \n
                    except Exception:
                        pass
                    self._line_open = True
                    # Reset _last_line_start to this line so next overwrite
                    # knows where to delete from.
                    self._last_line_start = self._safe_linestart()

            self.log_box.see(tk.END)
            self.log_box.configure(state="disabled")

        self.after(80, self._poll_log)

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        # Palette: BG: Deep Navy, CARD: Slate, ACC: Blue, HL: Pink/Red, FG: Off-white
        c = {"BG": "#1a1a2e", "CARD": "#16213e", "ACC": "#0f3460", "HL": "#e94560", "FG": "#eaeaea", "INPUT": "#0d1b2a"}
        self.colors = c
        
        # 1. Labels
        s.configure("TLabel", background=c["CARD"], foreground=c["FG"], font=("Consolas", 10))
        
        # 2. Textboxes (Entry) & Spinboxes
        # We apply the dark 'INPUT' color to the fieldbackground
        s.configure("TEntry", 
                    fieldbackground=c["INPUT"], 
                    foreground=c["FG"], 
                    insertcolor=c["FG"], 
                    borderwidth=0)
        
        s.configure("TSpinbox", 
                    fieldbackground=c["INPUT"], 
                    foreground=c["FG"], 
                    insertcolor=c["FG"], 
                    borderwidth=0)

        # 3. Buttons
        s.configure("TButton", background=c["ACC"], foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("TButton", background=[("active", c["HL"])])
        s.configure("Stop.TButton", background="#7a0020", foreground=c["FG"])
        
        # 4. Checkbutton - FIXED HOVER
        s.configure("TCheckbutton", background=c["CARD"], foreground=c["FG"], font=("Consolas", 10))
        s.map("TCheckbutton", 
              background=[("active", c["CARD"])],  # Keep background dark on hover
              foreground=[("active", c["HL"])])     # Turn text highlight color on hover

    def _build_ui(self):
        c = self.colors
        top = tk.Frame(self, bg=c["BG"], pady=10)
        top.pack(fill="x", padx=18)
        tk.Label(top, text="🔬 MICROPLASTICS TRAINER", bg=c["BG"], fg=c["HL"], font=("Consolas", 14, "bold")).pack(side="left")
        self.status_lbl = tk.Label(top, text="● Idle", bg=c["BG"], fg="#888", font=("Consolas", 10))
        self.status_lbl.pack(side="right")

        pane = tk.PanedWindow(self, orient="horizontal", bg=c["BG"], sashwidth=4)
        pane.pack(fill="both", expand=True, padx=12, pady=12)
        left, right = tk.Frame(pane, bg=c["BG"]), tk.Frame(pane, bg=c["BG"])
        pane.add(left, minsize=380); pane.add(right, minsize=450)

        def card(t, p):
            tk.Label(p, text=t, bg=c["BG"], fg=c["HL"], font=("Consolas", 9, "bold")).pack(anchor="w")
            f = tk.Frame(p, bg=c["CARD"], padx=10, pady=8); f.pack(fill="x", pady=(2, 8)); return f

        # FILES
        f1 = card("📁 FILES", left)
        self.yaml_var, self.model_var = tk.StringVar(), tk.StringVar(value="yolo26n.pt")
        for l, v in [("Dataset", self.yaml_var), ("Model", self.model_var)]:
            r = tk.Frame(f1, bg=c["CARD"]); r.pack(fill="x", pady=2)
            tk.Label(r, text=l, width=12, anchor="w", bg=c["CARD"], fg=c["FG"]).pack(side="left")
            ttk.Entry(r, textvariable=v).pack(side="left", fill="x", expand=True)
            ttk.Button(r, text="..", width=3, command=lambda var=v: var.set(filedialog.askopenfilename())).pack(side="left", padx=2)

        # PARAMETERS
        f2 = card("⚙️ PARAMETERS", left)
        self.ep_v, self.im_v, self.ba_v = tk.IntVar(value=epochs), tk.IntVar(value=imgsz), tk.IntVar(value=batch)
        self.lr0_v, self.lrf_v, self.wa_v, self.wo_v = tk.DoubleVar(value=0.001), tk.DoubleVar(value=0.01), tk.IntVar(value=3), tk.IntVar(value=6)
        self.conf_v = tk.DoubleVar(value=0.25)

        # Grid layout for parameters to save space
        params = [
            ("Epochs", self.ep_v), ("ImgSz", self.im_v),
            ("Batch", self.ba_v), ("Workers", self.wo_v),
            ("lr0", self.lr0_v), ("lrf", self.lrf_v),
            ("Warmup", self.wa_v), ("Conf", self.conf_v)
        ]
        for l, v in params:
            r = tk.Frame(f2, bg=c["CARD"]); r.pack(fill="x", pady=1)
            tk.Label(r, text=l, width=12, anchor="w", bg=c["CARD"], fg=c["FG"]).pack(side="left")
            if isinstance(v, tk.DoubleVar):
                ttk.Entry(r, textvariable=v, width=10).pack(side="left")
            else:
                ttk.Spinbox(r, textvariable=v, from_=1, to=2000, width=10).pack(side="left")

        # DEVICE
        f3 = card("💻 DEVICE", left)
        self.dev_v = tk.StringVar(value=device)
        for o in ["cpu", "cuda", "mps"]:
            tk.Radiobutton(f3, text=o.upper(), variable=self.dev_v, value=o, bg=c["CARD"], fg=c["FG"], selectcolor=c["ACC"]).pack(side="left", padx=5)

        # OPTIONS
        f4 = card("🔁 OPTIONS", left)
        self.res_v = tk.BooleanVar()
        ttk.Checkbutton(f4, text="Resume Training from last.pt", variable=self.res_v).pack(anchor="w")

        b_fr = tk.Frame(left, bg=c["BG"])
        b_fr.pack(fill="x", pady=5)
        self.train_btn = ttk.Button(b_fr, text="▶ START TRAINING", command=self.start_training)
        self.stop_btn  = ttk.Button(b_fr, text="■ STOP", style="Stop.TButton", command=self.stop_training, state="disabled")
        self.test_btn  = ttk.Button(b_fr, text="🔍 TEST IMAGE", command=self.run_test_image)
        self.sim_btn   = ttk.Button(b_fr, text="⚡ SIMULATE LOG", command=self.simulate_progress)
        self.clr_btn   = ttk.Button(b_fr, text="🗑 CLEAR LOG", command=self.clear_log)
        self.train_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.stop_btn.pack(side="left",  expand=True, fill="x", padx=2)
        self.test_btn.pack(side="left",  expand=True, fill="x", padx=2)
        self.sim_btn.pack(side="left",   expand=True, fill="x", padx=2)
        self.clr_btn.pack(side="left",   expand=True, fill="x", padx=2)

        # Log
        tk.Label(right, text="📋 TRAINING LOG", bg=c["BG"], fg=c["HL"], font=("Consolas", 9, "bold")).pack(anchor="w")
        self.log_box = tk.Text(right, state="disabled", bg="#0d1b2a", fg="#c8ffc8", font=("Consolas", 9), wrap="none")
        self.log_box.pack(fill="both", expand=True)

    def set_status(self, t, c): self.after(0, lambda: self.status_lbl.configure(text=f"● {t}", fg=c))
    def training_finished(self): self.after(0, lambda: [self.train_btn.configure(state="normal"), self.stop_btn.configure(state="disabled")])
    def stop_training(self): self.cleanup_subprocess(); self.set_status("Stopping...", "orange")

    def clear_log(self):
        """Clear the log display."""
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")
        self._last_line_start = None
        self._ansi_in_progress = False
        self._pending_r = False
        self._log_partial = ""
        self._line_open = False

    def start_training(self):
        y, m = self.yaml_var.get(), self.model_var.get()
        if not y or not os.path.exists(y):
            print("[ERROR] Dataset YAML not found.")
            return
        self.train_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        threading.Thread(target=_run_training, args=(
            y, m, self.ep_v.get(), self.im_v.get(), self.ba_v.get(), 
            self.dev_v.get(), self.wo_v.get(), self.lr0_v.get(), 
            self.lrf_v.get(), self.wa_v.get(), self.res_v.get(), self
        ), daemon=True).start()

    def run_test_image(self):
        """Launch test image inference in a background thread."""
        threading.Thread(
            target=_run_test_image,
            args=(self, float(self.conf_v.get())),
            daemon=True,
        ).start()

    def simulate_progress(self):
        """Emit ANSI-colored progress updates with '\r' into the log_queue for testing."""
        def _sim():
            for i in range(0, 101, 5):
                # green text with reset, using CR to overwrite
                chunk = f"\r\x1b[32mSimulated: {i}%\x1b[0m"
                log_queue.put(chunk)
                time.sleep(0.12)
            # finish with newline
            log_queue.put("\n")
            log_queue.put("Simulation complete.\n")

        threading.Thread(target=_sim, daemon=True).start()

if __name__ == "__main__":
    app = TrainerApp()
    app.mainloop()