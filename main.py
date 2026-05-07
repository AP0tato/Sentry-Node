from ultralytics import YOLO
import time
from collections import Counter
import torch
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import glob

# ── Device detection ──────────────────────────────────────────────────────────
device = "cpu"
batch  = 8
epochs = 50
imgsz  = 1280

if torch.cuda.is_available():
    batch  = 4
    epochs = 100
    imgsz  = 1280
    device = "cuda"
elif torch.backends.mps.is_available():
    batch  = 8
    epochs = 100
    imgsz  = 1280
    device = "mps"

# ── Hardware stub functions ────────────────────────────────────────────────────
def move_servo(angle):
    print("[SERVO] Rotate", angle, "degrees")

def pump(delay):
    print("[PUMP] Moves stepper forward for", delay, "seconds")
    time.sleep(delay)

def take_photo():
    print("[Camera] Taking photo")
    time.sleep(1)

# ── Redirect stdout/stderr into Tkinter log box ───────────────────────────────
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.after(0, self._append, text)

    def _append(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def flush(self):
        pass

# ── Training thread wrapper ────────────────────────────────────────────────────
training_thread = None
stop_flag       = False
current_model   = None          # holds the YOLO instance during training

def _run_training(yaml_path, model_path, ep, img, bat, dev, resume, app):
    global stop_flag, current_model
    stop_flag = False

    try:
        app.set_status("Training…", "yellow")
        yolo = YOLO(model_path)
        current_model = yolo

        train_kwargs = dict(
            data          = yaml_path,
            epochs        = ep,
            imgsz         = img,
            batch         = bat,
            device        = dev,
            workers       = 6,
            cache         = False,
            lr0           = 0.01,
            lrf           = 0.1,
            warmup_epochs = 3,
            name          = "microplastics_run",
        )
        if resume:
            results = yolo.train(resume=True)
        else:
            results = yolo.train(**train_kwargs)

        if results is not None:
            rd = results.results_dict
            print("\n─── Final Metrics ───────────────────────")
            print(f"  mAP50:     {rd.get('metrics/mAP50(B)',    0):.4f}")
            print(f"  mAP50-95:  {rd.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"  Precision: {rd.get('metrics/precision(B)', 0):.4f}")
            print(f"  Recall:    {rd.get('metrics/recall(B)',    0):.4f}")
            print("─────────────────────────────────────────\n")

        app.set_status("Done ✓", "lime green")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        app.set_status("Error — see log", "red")
    finally:
        current_model = None
        app.training_finished()

# ── Main GUI class ─────────────────────────────────────────────────────────────
class TrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Microplastics YOLO Trainer")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)
        self.minsize(760, 620)

        self._build_styles()
        self._build_ui()

        # redirect stdout/stderr → log box
        redirector = TextRedirector(self.log_box)
        sys.stdout = redirector
        sys.stderr = redirector

        print(f"[INFO] Device detected: {device.upper()}")
        print(f"[INFO] Default  epochs={epochs}  imgsz={imgsz}  batch={batch}\n")

    # ── Styles ──────────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        BG   = "#1a1a2e"
        CARD = "#16213e"
        ACC  = "#0f3460"
        HL   = "#e94560"
        FG   = "#eaeaea"

        style.configure("Card.TFrame",       background=CARD)
        style.configure("TLabel",            background=CARD,  foreground=FG,  font=("Consolas", 10))
        style.configure("Header.TLabel",     background=BG,    foreground=HL,  font=("Consolas", 13, "bold"))
        style.configure("Sub.TLabel",        background=CARD,  foreground="#aaaaaa", font=("Consolas", 9))

        style.configure("TEntry",            fieldbackground="#0d1b2a", foreground=FG,
                        insertcolor=FG, font=("Consolas", 10))
        style.configure("TSpinbox",          fieldbackground="#0d1b2a", foreground=FG,
                        insertcolor=FG, font=("Consolas", 10))

        style.configure("TButton",           background=ACC, foreground=FG,
                        font=("Consolas", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", HL)],
                  foreground=[("active", "#ffffff")])

        style.configure("Stop.TButton",      background="#7a0020", foreground=FG,
                        font=("Consolas", 10, "bold"), padding=6)
        style.map("Stop.TButton",
                  background=[("active", "#ff002b")])

        style.configure("TCheckbutton",      background=CARD, foreground=FG,
                        font=("Consolas", 10))
        style.map("TCheckbutton",            background=[("active", CARD)])

        self.colors = dict(BG=BG, CARD=CARD, ACC=ACC, HL=HL, FG=FG)

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = self.colors
        self.configure(bg=c["BG"])

        # ── title bar
        title_bar = tk.Frame(self, bg=c["BG"], pady=10)
        title_bar.pack(fill="x", padx=18)
        tk.Label(title_bar, text="🔬 MICROPLASTICS YOLO TRAINER",
                 bg=c["BG"], fg=c["HL"],
                 font=("Consolas", 15, "bold")).pack(side="left")
        self.status_lbl = tk.Label(title_bar, text="● Idle",
                                   bg=c["BG"], fg="#888888",
                                   font=("Consolas", 10))
        self.status_lbl.pack(side="right")

        # ── main paned window (config left | log right)
        pane = tk.PanedWindow(self, orient="horizontal",
                              bg=c["BG"], sashwidth=6, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left  = tk.Frame(pane, bg=c["BG"])
        right = tk.Frame(pane, bg=c["BG"])
        pane.add(left,  minsize=320)
        pane.add(right, minsize=360)

        self._build_config_panel(left)
        self._build_log_panel(right)

    def _build_config_panel(self, parent):
        c = self.colors

        def card(p, title):
            wrapper = tk.Frame(p, bg=c["BG"])
            wrapper.pack(fill="x", pady=(0, 10))
            tk.Label(wrapper, text=title, bg=c["BG"], fg=c["HL"],
                     font=("Consolas", 10, "bold")).pack(anchor="w", pady=(0, 4))
            frm = tk.Frame(wrapper, bg=c["CARD"], padx=12, pady=10)
            frm.pack(fill="x")
            return frm

        def row(frm, label, widget_factory):
            r = tk.Frame(frm, bg=c["CARD"])
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, width=14, anchor="w",
                     bg=c["CARD"], fg=c["FG"],
                     font=("Consolas", 10)).pack(side="left")
            w = widget_factory(r)
            w.pack(side="left", fill="x", expand=True)
            return w

        def browse_row(frm, label, var, filetypes, default=""):
            r = tk.Frame(frm, bg=c["CARD"])
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, width=14, anchor="w",
                     bg=c["CARD"], fg=c["FG"],
                     font=("Consolas", 10)).pack(side="left")
            ent = ttk.Entry(r, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            btn = ttk.Button(r, text="…", width=3,
                             command=lambda: self._browse(var, filetypes))
            btn.pack(side="left", padx=(4, 0))
            if default:
                var.set(default)

        # ── Files card
        files_card = card(parent, "📁  FILES")

        self.yaml_var  = tk.StringVar()
        self.model_var = tk.StringVar()
        browse_row(files_card, "Dataset YAML", self.yaml_var,
                   [("YAML files", "*.yaml"), ("All", "*.*")])
        browse_row(files_card, "Model (.pt)",  self.model_var,
                   [("PyTorch weights", "*.pt"), ("All", "*.*")],
                   default="yolov8s.pt")

        # ── Hyperparams card
        hp_card = card(parent, "⚙️  HYPERPARAMETERS")

        self.epochs_var = tk.IntVar(value=epochs)
        self.imgsz_var  = tk.IntVar(value=imgsz)
        self.batch_var  = tk.IntVar(value=batch)
        self.lr0_var    = tk.DoubleVar(value=0.01)
        self.lrf_var    = tk.DoubleVar(value=0.1)
        self.warm_var   = tk.IntVar(value=3)
        self.work_var   = tk.IntVar(value=2)

        def spinbox(p, var, from_, to, inc=1):
            return ttk.Spinbox(p, textvariable=var,
                               from_=from_, to=to, increment=inc, width=10)

        row(hp_card, "Epochs",      lambda p: spinbox(p, self.epochs_var, 1,   500))
        row(hp_card, "Image size",  lambda p: spinbox(p, self.imgsz_var,  320, 1920, 32))
        row(hp_card, "Batch size",  lambda p: spinbox(p, self.batch_var,  1,   64))
        row(hp_card, "lr0",         lambda p: ttk.Entry(p, textvariable=self.lr0_var, width=10))
        row(hp_card, "lrf",         lambda p: ttk.Entry(p, textvariable=self.lrf_var, width=10))
        row(hp_card, "Warmup epoc", lambda p: spinbox(p, self.warm_var,   0,   20))
        row(hp_card, "Workers",     lambda p: spinbox(p, self.work_var,   0,   16))

        # ── Device card
        dev_card = card(parent, "💻  DEVICE")
        self.device_var = tk.StringVar(value=device)
        dev_row = tk.Frame(dev_card, bg=c["CARD"])
        dev_row.pack(fill="x")
        for d in ["cpu", "cuda", "mps"]:
            tk.Radiobutton(dev_row, text=d.upper(), variable=self.device_var, value=d,
                           bg=c["CARD"], fg=c["FG"], selectcolor=c["ACC"],
                           activebackground=c["CARD"], activeforeground=c["HL"],
                           font=("Consolas", 10)).pack(side="left", padx=8)

        # ── Resume checkbox
        opt_card = card(parent, "🔁  OPTIONS")
        self.resume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_card, text="Resume from last checkpoint (last.pt)",
                        variable=self.resume_var).pack(anchor="w")
        tk.Label(opt_card,
                 text="When resuming, model path must point to last.pt",
                 bg=c["CARD"], fg="#888888",
                 font=("Consolas", 8)).pack(anchor="w")

        # ── Buttons
        btn_frame = tk.Frame(parent, bg=c["BG"])
        btn_frame.pack(fill="x", pady=(4, 0))

        self.train_btn  = ttk.Button(btn_frame, text="▶  TRAIN",
                                     command=self.start_training)
        self.resume_btn = ttk.Button(btn_frame, text="⟳  RESUME",
                                     command=self.start_resume)
        self.stop_btn   = ttk.Button(btn_frame, text="■  STOP",
                                     style="Stop.TButton",
                                     command=self.stop_training,
                                     state="disabled")

        self.train_btn.pack(side="left",  expand=True, fill="x", padx=(0, 4))
        self.resume_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.stop_btn.pack(side="left",   expand=True, fill="x")

    def _build_log_panel(self, parent):
        c = self.colors
        tk.Label(parent, text="📋  TRAINING LOG",
                 bg=c["BG"], fg=c["HL"],
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(0, 4))

        log_frame = tk.Frame(parent, bg=c["CARD"])
        log_frame.pack(fill="both", expand=True)

        self.log_box = tk.Text(
            log_frame,
            state="disabled",
            bg="#0d1b2a",
            fg="#c8ffc8",
            font=("Consolas", 9),
            wrap="word",
            insertbackground="#c8ffc8",
            relief="flat",
            padx=8, pady=8,
            selectbackground="#0f3460",
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True)

        # clear button
        ttk.Button(parent, text="🗑  Clear log",
                   command=self._clear_log).pack(anchor="e", pady=(6, 0))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _browse(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def set_status(self, text, color):
        self.status_lbl.after(0, lambda: self.status_lbl.configure(
            text=f"● {text}", fg=color))

    def training_finished(self):
        self.after(0, self._enable_buttons)

    def _enable_buttons(self):
        self.train_btn.configure(state="normal")
        self.resume_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _disable_buttons(self):
        self.train_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def stop_training(self):
        global stop_flag, current_model
        stop_flag = True
        print("\n[INFO] Stop requested — will halt after current epoch saves.\n")
        # Ultralytics respects KeyboardInterrupt in the trainer loop
        if current_model and hasattr(current_model, "trainer") and current_model.trainer:
            current_model.trainer.stop = True
        self.set_status("Stopping…", "orange")

    # ── Training launchers ────────────────────────────────────────────────────
    def _validate(self):
        yaml = self.yaml_var.get().strip()
        model_path = self.model_var.get().strip()
        if not yaml:
            print("[ERROR] Please select a dataset YAML file.\n")
            return None, None
        if not os.path.exists(yaml):
            print(f"[ERROR] YAML not found: {yaml}\n")
            return None, None
        if not model_path:
            print("[ERROR] Please select a model .pt file.\n")
            return None, None
        return yaml, model_path

    def start_training(self):
        yaml, model_path = self._validate()
        if yaml is None:
            return
        self._disable_buttons()
        self.set_status("Training…", "yellow")
        t = threading.Thread(
            target=_run_training,
            args=(yaml, model_path,
                  self.epochs_var.get(), self.imgsz_var.get(),
                  self.batch_var.get(),  self.device_var.get(),
                  False, self),
            daemon=True,
        )
        t.start()

    def start_resume(self):
        model_path = self.model_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            # try to find latest last.pt automatically
            checkpoints = glob.glob("runs/detect/*/weights/last.pt")
            if checkpoints:
                model_path = max(checkpoints, key=os.path.getmtime)
                self.model_var.set(model_path)
                print(f"[INFO] Auto-found checkpoint: {model_path}\n")
            else:
                print("[ERROR] No last.pt found. Train first or select it manually.\n")
                return

        self._disable_buttons()
        self.set_status("Resuming…", "yellow")
        t = threading.Thread(
            target=_run_training,
            args=(self.yaml_var.get(), model_path,
                  self.epochs_var.get(), self.imgsz_var.get(),
                  self.batch_var.get(),  self.device_var.get(),
                  True, self),
            daemon=True,
        )
        t.start()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TrainerApp()
    app.mainloop()