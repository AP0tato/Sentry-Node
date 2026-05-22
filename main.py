"""
Microplastics YOLO Trainer
──────────────────────────
GUI front-end for training YOLO models on microplastics datasets.
"""

import atexit
import gc
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

import cv2
import torch
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO


# ── Environment ───────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ── Device detection ──────────────────────────────────────────────────────────
def _detect_device():
    if torch.cuda.is_available():
        return "cuda", 4, 100, 1280
    if torch.backends.mps.is_available():
        return "mps", 8, 100, 1280
    return "cpu", 8, 50, 1280

DEVICE, BATCH, EPOCHS, IMGSZ = _detect_device()

# ── Global process handle ─────────────────────────────────────────────────────
_current_process: "subprocess.Popen | None" = None
_stop_flag = False
_stop_event = threading.Event()
_resume_flag = False

# ══════════════════════════════════════════════════════════════════════════════
#  Dataset utilities
# ══════════════════════════════════════════════════════════════════════════════

def _repair_label_dir(label_dir: str, nc: int) -> tuple[int, int, int]:
    files_changed = 0
    segments_fixed = 0
    lines_removed = 0

    for txt_path in Path(label_dir).glob("*.txt"):
        try:
            raw_lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            print(f"[FIX]   Cannot read {txt_path.name}: {exc}", flush=True)
            continue

        new_lines: list[str] = []
        changed = False

        for line in raw_lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                lines_removed += 1
                changed = True
                continue

            if cls_id < 0 or cls_id >= nc:
                lines_removed += 1
                changed = True
                continue

            if len(parts) < 5:
                lines_removed += 1
                changed = True
                continue

            if len(parts) > 5:
                parts = parts[:5]
                segments_fixed += 1
                changed = True

            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                lines_removed += 1
                changed = True
                continue

            if any(v < 0.0 or v > 1.0 for v in coords):
                lines_removed += 1
                changed = True
                continue

            new_lines.append(f"{cls_id} " + " ".join(parts[1:]))

        if changed:
            txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                encoding="utf-8")
            files_changed += 1

    return files_changed, segments_fixed, lines_removed


def _delete_caches(dataset_dir: str) -> int:
    deleted = 0
    for cache in Path(dataset_dir).rglob("*.cache"):
        try:
            cache.unlink()
            print(f"[FIX] Deleted cache: {cache}", flush=True)
            deleted += 1
        except Exception as exc:
            print(f"[FIX] Could not delete {cache}: {exc}", flush=True)
    return deleted


def _run_fix_dataset(yaml_path: str, app: "TrainerApp") -> None:
    try:
        app.set_status("Fixing dataset…", "yellow")

        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        nc: int = int(data.get("nc", 0))
        dataset_dir = os.path.dirname(yaml_path)

        print(f"\n[FIX] Dataset : {dataset_dir}", flush=True)
        print(f"[FIX] Classes : {nc}\n", flush=True)

        total_files = total_segs = total_removed = 0

        for split in ("train", "valid", "test"):
            label_dir = os.path.join(dataset_dir, split, "labels")
            if not os.path.isdir(label_dir):
                continue

            n_txt = len(list(Path(label_dir).glob("*.txt")))
            print(f"[FIX] Scanning {split}: {n_txt} label files…", flush=True)
            fc, sf, lr = _repair_label_dir(label_dir, nc)
            total_files   += fc
            total_segs    += sf
            total_removed += lr

        cache_count = _delete_caches(dataset_dir)

        print(f"\n[FIX] ── Summary ─────────────────────────", flush=True)
        print(f"[FIX]   Files modified        : {total_files}", flush=True)
        print(f"[FIX]   Segment→bbox fixed    : {total_segs}", flush=True)
        print(f"[FIX]   Bad lines removed     : {total_removed}", flush=True)
        print(f"[FIX]   Cache files deleted   : {cache_count}", flush=True)
        print(f"[FIX] ─────────────────────────────────────", flush=True)
        print(f"[FIX] Dataset is ready — you can now train.\n", flush=True)

        app.set_status("Dataset fixed ✓", "lime green")

    except Exception as exc:
        print(f"\n[ERROR] Fix dataset failed: {exc}\n", flush=True)
        app.set_status("Error — see log", "red")


def _run_clean_dataset(yaml_path: str, keep_classes: set[str],
                       app: "TrainerApp") -> None:
    try:
        app.set_status("Cleaning dataset…", "yellow")

        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        all_names: list[str] = data["names"]
        kept_indices = [i for i, n in enumerate(all_names) if n in keep_classes]
        old_to_new   = {old: new for new, old in enumerate(kept_indices)}
        new_names    = [all_names[i] for i in kept_indices]
        new_nc       = len(new_names)

        print(f"\n[CLEAN] Keeping {new_nc} classes:", flush=True)
        for i, n in enumerate(new_names):
            print(f"  [{i}] {n}", flush=True)
        print(flush=True)

        dataset_dir = os.path.dirname(yaml_path)
        total_removed = 0

        for split in ("train", "valid", "test"):
            label_dir = os.path.join(dataset_dir, split, "labels")
            if not os.path.isdir(label_dir):
                continue

            for txt_path in Path(label_dir).glob("*.txt"):
                try:
                    raw = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue

                new_lines: list[str] = []
                for line in raw:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        total_removed += 1
                        continue
                    if cls_id not in old_to_new:
                        total_removed += 1
                        continue
                    parts[0] = str(old_to_new[cls_id])
                    new_lines.append(" ".join(parts))

                txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                    encoding="utf-8")

            _repair_label_dir(label_dir, new_nc)

        data["nc"]    = new_nc
        data["names"] = new_names
        with open(yaml_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)

        _delete_caches(dataset_dir)

        print(f"[CLEAN] Removed {total_removed} label entries.", flush=True)
        print(f"[CLEAN] Updated data.yaml → {new_nc} classes.\n", flush=True)
        app.set_status("Dataset cleaned ✓", "lime green")

    except Exception as exc:
        print(f"\n[ERROR] Dataset clean failed: {exc}\n", flush=True)
        app.set_status("Error — see log", "red")


def _run_oversample(yaml_path: str, target_ratio: float,
                    app: "TrainerApp") -> None:
    try:
        app.set_status("Oversampling…", "yellow")

        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        dataset_dir = os.path.dirname(yaml_path)
        label_dir   = os.path.join(dataset_dir, "train", "labels")
        image_dir   = os.path.join(dataset_dir, "train", "images")

        if not os.path.isdir(label_dir) or not os.path.isdir(image_dir):
            print("[OVERSAMPLE] train/labels or train/images directory not found.\n", flush=True)
            app.set_status("Error — see log", "red")
            return

        class_to_files: dict[int, list[Path]] = {}
        for txt in Path(label_dir).glob("*.txt"):
            classes_in_file: set[int] = set()
            for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_file.add(int(float(parts[0])))
                    except ValueError:
                        pass
            for cls_id in classes_in_file:
                class_to_files.setdefault(cls_id, []).append(txt)

        if not class_to_files:
            print("[OVERSAMPLE] No annotated label files found in train/labels.\n", flush=True)
            app.set_status("Error — see log", "red")
            return

        max_count = max(len(v) for v in class_to_files.values())
        target    = int(max_count * target_ratio)
        names     = data.get("names", [])

        print(f"\n[OVERSAMPLE] Most-common class count : {max_count}", flush=True)
        print(f"[OVERSAMPLE] Target per-class count  : {target}  ({target_ratio*100:.0f}% of max)\n", flush=True)

        total_copies = 0
        for cls_id, files in sorted(class_to_files.items()):
            current = len(files)
            needed  = max(0, target - current)
            name    = names[cls_id] if cls_id < len(names) else str(cls_id)

            if needed == 0:
                print(f"  [{name}]  {current} samples — no copies needed", flush=True)
                continue

            print(f"  [{name}]  {current} samples → adding {needed} copies", flush=True)

            for i in range(needed):
                src_lbl = files[i % len(files)]
                stem    = src_lbl.stem

                src_img: "Path | None" = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                    candidate = Path(image_dir) / (stem + ext)
                    if candidate.exists():
                        src_img = candidate
                        break

                if src_img is None:
                    continue

                new_stem = f"{stem}_os{i:05d}"
                dst_lbl  = Path(label_dir) / (new_stem + ".txt")
                dst_img  = Path(image_dir) / (new_stem + src_img.suffix)

                shutil.copy2(src_lbl, dst_lbl)
                shutil.copy2(src_img, dst_img)
                total_copies += 1

        _delete_caches(dataset_dir)
        print(f"\n[OVERSAMPLE] Done — {total_copies} image+label pairs copied.\n", flush=True)
        app.set_status("Oversample done ✓", "lime green")

    except Exception as exc:
        print(f"\n[ERROR] Oversample failed: {exc}\n", flush=True)
        app.set_status("Error — see log", "red")


# ══════════════════════════════════════════════════════════════════════════════
#  Training worker
# ══════════════════════════════════════════════════════════════════════════════

def _run_training(yaml_path: str, model_path: str,
                  ep: int, img: int, bat: int, dev: str,
                  workers: int, lr0: float, lrf: float,
                  warmup: int, cache_mode: str,
                  cls_gain: float, box_gain: float,
                  app: "TrainerApp", checkpoint_path: str = "") -> None:
    global _stop_flag, _resume_flag
    _stop_flag = False
    _stop_event.clear()

    try:
        app.set_status("Training…", "yellow")

        print(f"[INFO] Device    : {dev.upper()}", flush=True)
        print(f"[INFO] epochs={ep}  imgsz={img}  batch={bat}", flush=True)
        print(f"[INFO] cache={cache_mode}", flush=True)
        print(f"[INFO] Starting training via Python API", flush=True)
        print(f"[INFO] cls={cls_gain}  box={box_gain}\n", flush=True)

        cache_arg = False if cache_mode == "none" else cache_mode

        if _resume_flag:
            print(f"[INFO] Resuming training from checkpoint\n", flush=True)
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                print(f"[ERROR] Checkpoint file not selected or not found\n", flush=True)
                app.set_status("Error — checkpoint not selected", "red")
                _resume_flag = False
                app.training_finished()
                return
            
            model = YOLO(checkpoint_path)
            model.train(
                resume=True,
            )
        else:
            model = YOLO(model_path)
            model.train(
                data=yaml_path,
                epochs=ep,
                imgsz=img,
                batch=bat,
                device=dev,
                workers=workers,
                lr0=lr0,
                lrf=lrf,
                warmup_epochs=float(warmup),
                cls=cls_gain,
                box=box_gain,
                name="microplastics_run",
                cache=cache_arg,
            )

        if _stop_flag:
            app.set_status("Stopped", "orange")
        else:
            app.set_status("Done ✓", "lime green")
            _run_test_image(app, conf=float(app.conf_v.get()), iou=float(app.iou_v.get()), auto=True)

    except Exception as exc:
        if _stop_flag:
            app.set_status("Stopped", "orange")
        else:
            print(f"\n[ERROR] Training failed: {exc}\n", flush=True)
            app.set_status("Error — see log", "red")
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _resume_flag = False
        app.training_finished()


# ══════════════════════════════════════════════════════════════════════════════
#  Inference worker
# ══════════════════════════════════════════════════════════════════════════════

def _run_test_image(app: "TrainerApp", conf: float = 0.25, iou: float = 0.45,
                    auto: bool = False, test_img_path: str = "") -> None:
    try:
        if not auto:
            app.set_status("Running inference…", "yellow")

        test_img = test_img_path or os.path.join(os.getcwd(), "test_image.jpg")
        if not os.path.exists(test_img):
            print("[ERROR] test_image.jpg not found in the current directory.\n", flush=True)
            if not auto:
                app.set_status("Idle", "#888")
            return

        checkpoints = glob.glob("runs/detect/*/weights/best.pt")
        if not checkpoints:
            print("[ERROR] No best.pt found — train the model first.\n", flush=True)
            if not auto:
                app.set_status("Idle", "#888")
            return

        weight_path = max(checkpoints, key=os.path.getmtime)
        print(f"\n[INFO] Weights : {weight_path}", flush=True)
        print(f"[INFO] Image   : {test_img}", flush=True)
        print(f"[INFO] Conf    : {conf}", flush=True)
        print(f"[INFO] IOU     : {iou}\n", flush=True)

        model   = YOLO(weight_path)
        results = model.predict(test_img, conf=conf, iou=iou, verbose=True)

        if not results:
            print("[ERROR] No results returned.\n", flush=True)
            if not auto:
                app.set_status("Idle", "#888")
            return

        res = results[0]
        # Applied the show_labels toggle
        annotated = res.plot(labels=app.show_labels_v.get())

        boxes = res.boxes if res.boxes is not None else []
        det_names = [res.names[int(b.cls)] for b in boxes]
        counts = Counter(det_names)
        print("─── Detection Summary ───────────────────", flush=True)
        print(f"  Total detected : {len(det_names)}", flush=True)
        if det_names:
            for cls, n in counts.most_common():
                pct = n / len(det_names) * 100
                print(f"  {cls:<25} {n:>3}  ({pct:.1f}%)", flush=True)
        else:
            print("  No objects detected.", flush=True)
        print("─────────────────────────────────────────\n", flush=True)

        h, w  = annotated.shape[:2]
        scale = min(1200 / w, 900 / h, 1.0)
        disp  = cv2.resize(annotated, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detections", int(w * scale), int(h * scale))
        cv2.imshow("Detections", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not auto:
            app.set_status("Done ✓", "lime green")

    except Exception as exc:
        print(f"\n[ERROR] Inference failed: {exc}\n", flush=True)
        if not auto:
            app.set_status("Error — see log", "red")


# ══════════════════════════════════════════════════════════════════════════════
#  Clean Dataset dialog
# ══════════════════════════════════════════════════════════════════════════════

_JUNK_DEFAULTS = {
    "Microbead", "Insect matter", "Easylift tab",
    "Anthropogenic", "Salt", "Unknown", "Natural material",
}


class CleanDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, yaml_path: str, app: "TrainerApp"):
        super().__init__(parent)
        self.title("Clean Dataset — Select Classes to KEEP")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)
        self.grab_set()

        self.yaml_path = yaml_path
        self.app = app

        c = _COLORS

        try:
            with open(yaml_path) as fh:
                data = yaml.safe_load(fh)
            names: list[str] = data.get("names", [])
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read YAML:\n{exc}")
            self.destroy()
            return

        tk.Label(self,
                 text="✅  Tick classes to KEEP  (untick = remove)",
                 bg=c["BG"], fg=c["HL"],
                 font=("Consolas", 10, "bold")).pack(padx=16, pady=(12, 4))

        canvas = tk.Canvas(self, bg=c["CARD"], highlightthickness=0,
                           width=380, height=400)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 4), pady=8)
        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=8)

        inner = tk.Frame(canvas, bg=c["CARD"])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        self.vars: dict[str, tk.BooleanVar] = {}
        for name in names:
            var = tk.BooleanVar(value=(name not in _JUNK_DEFAULTS))
            self.vars[name] = var
            tk.Checkbutton(
                inner, text=name, variable=var,
                bg=c["CARD"], fg=c["FG"], selectcolor=c["ACC"],
                activebackground=c["CARD"], activeforeground=c["HL"],
                font=("Consolas", 9),
            ).pack(anchor="w", padx=8, pady=1)

        btn_fr = tk.Frame(self, bg=c["BG"])
        btn_fr.pack(fill="x", padx=12, pady=(4, 12))
        ttk.Button(btn_fr, text="✅  Apply & Clean",
                   command=self._apply).pack(side="left", expand=True,
                                             fill="x", padx=(0, 4))
        ttk.Button(btn_fr, text="✖  Cancel",
                   command=self.destroy).pack(side="left", expand=True, fill="x")

    def _apply(self):
        keep = {name for name, var in self.vars.items() if var.get()}
        if not keep:
            messagebox.showwarning("Nothing selected",
                                   "Select at least one class to keep.")
            return
        self.destroy()
        threading.Thread(
            target=_run_clean_dataset,
            args=(self.yaml_path, keep, self.app),
            daemon=True,
        ).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Colour palette
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "BG":    "#1a1a2e",
    "CARD":  "#16213e",
    "ACC":   "#0f3460",
    "HL":    "#e94560",
    "FG":    "#eaeaea",
    "INPUT": "#0d1b2a",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Main GUI
# ══════════════════════════════════════════════════════════════════════════════

class TrainerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Microplastics YOLO Trainer")
        self.configure(bg=_COLORS["BG"])
        self.minsize(500, 720)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._build_styles()
        self._build_ui()

        atexit.register(self._cleanup_subprocess)

        print(f"[INFO] Device : {DEVICE.upper()}", flush=True)
        print(f"[INFO] epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}\n", flush=True)

    def _on_closing(self):
        self._cleanup_subprocess()
        self.destroy()
        sys.exit(0)

    def _cleanup_subprocess(self):
        global _current_process
        if _current_process and _current_process.poll() is None:
            try:
                _current_process.terminate()
                _current_process.wait(timeout=2)
            except Exception:
                try:
                    _current_process.kill()
                except Exception:
                    pass
        _current_process = None

    def set_status(self, text: str, color: str):
        self.after(0, lambda: self.status_lbl.configure(
            text=f"● {text}", fg=color))

    def training_finished(self):
        self.after(0, lambda: [
            self.train_btn.configure(state="normal"),
            self.stop_btn.configure(state="disabled"),
        ])

    def start_training(self):
        y = self.yaml_var.get().strip()
        m = self.model_var.get().strip()
        if not y or not os.path.exists(y):
            print("[ERROR] Dataset YAML not found — select it first.\n", flush=True)
            return
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(
            target=_run_training,
            args=(y, m,
                  self.ep_v.get(), self.im_v.get(), self.ba_v.get(),
                  self.dev_v.get(), self.wo_v.get(),
                  float(self.lr0_v.get()), float(self.lrf_v.get()),
                  self.wa_v.get(), self.cache_v.get(),
                  float(self.cls_v.get()), float(self.box_v.get()),
                  self, self.checkpoint_var.get()),
            daemon=True,
        ).start()

    def resume_training(self):
        global _resume_flag
        _resume_flag = True
        self.start_training()

    def stop_training(self):
        global _stop_flag
        _stop_flag = True
        _stop_event.set()
        self._cleanup_subprocess()
        self.set_status("Stopping…", "orange")

    def run_test_image(self):
        image_path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All", "*.*")],
        )
        if not image_path:
            return
        threading.Thread(
            target=_run_test_image,
            args=(self, float(self.conf_v.get()), float(self.iou_v.get()), False, image_path),
            daemon=True,
        ).start()

    def fix_dataset(self):
        y = self.yaml_var.get().strip()
        if not y or not os.path.exists(y):
            print("[ERROR] Select a dataset YAML first.\n", flush=True)
            return
        threading.Thread(
            target=_run_fix_dataset,
            args=(y, self),
            daemon=True,
        ).start()

    def open_clean_dialog(self):
        y = self.yaml_var.get().strip()
        if not y or not os.path.exists(y):
            print("[ERROR] Select a dataset YAML first.\n", flush=True)
            return
        CleanDialog(self, y, self)

    def oversample_dataset(self):
        y = self.yaml_var.get().strip()
        if not y or not os.path.exists(y):
            print("[ERROR] Select a dataset YAML first.\n", flush=True)
            return
        threading.Thread(
            target=_run_oversample,
            args=(y, 0.30, self),
            daemon=True,
        ).start()

    def _build_styles(self):
        c = _COLORS
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TLabel", background=c["CARD"], foreground=c["FG"], font=("Consolas", 10))
        s.configure("TEntry", fieldbackground=c["INPUT"], foreground=c["FG"], insertcolor=c["FG"], borderwidth=0)
        s.configure("TSpinbox", fieldbackground=c["INPUT"], foreground=c["FG"], insertcolor=c["FG"], borderwidth=0)
        
        s.configure("TButton", background=c["ACC"], foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("TButton", background=[("active", c["HL"])])

        s.configure("Stop.TButton", background="#7a0020", foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("Stop.TButton", background=[("active", "#ff002b")])

        s.configure("Resume.TButton", background="#1a4a6a", foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("Resume.TButton", background=[("active", "#2d7aaa")])

        s.configure("Fix.TButton", background="#1a4a1a", foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("Fix.TButton", background=[("active", "#2d7a2d")])

        s.configure("OS.TButton", background="#2a2a6a", foreground=c["FG"], font=("Consolas", 10, "bold"), borderwidth=0)
        s.map("OS.TButton", background=[("active", "#4444cc")])

        s.configure("TCheckbutton", background=c["CARD"], foreground=c["FG"], font=("Consolas", 10))
        s.map("TCheckbutton", background=[("active", c["CARD"])], foreground=[("active", c["HL"])])

        s.configure("TRadiobutton", background=c["CARD"], foreground=c["FG"], font=("Consolas", 10))

        # ── Styled Combobox for Cache selection ──
        s.configure("TCombobox",
                    fieldbackground=c["INPUT"], 
                    background=c["ACC"], 
                    foreground=c["FG"],
                    insertcolor=c["FG"], 
                    borderwidth=0)
        
        s.configure("Custom.TCombobox", 
                    fieldbackground=c["INPUT"], 
                    foreground=c["FG"], 
                    background=c["ACC"], 
                    borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", c["INPUT"])
        self.option_add("*TCombobox*Listbox.foreground", c["FG"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["HL"])
        self.option_add("*TCombobox*Listbox.selectForeground", c["FG"])

    def _build_ui(self):
        c = _COLORS
        top = tk.Frame(self, bg=c["BG"], pady=10)
        top.pack(fill="x", padx=18)
        tk.Label(top, text="🔬  MICROPLASTICS TRAINER", bg=c["BG"], fg=c["HL"], font=("Consolas", 14, "bold")).pack(side="left")
        self.status_lbl = tk.Label(top, text="● Idle", bg=c["BG"], fg="#888", font=("Consolas", 10))
        self.status_lbl.pack(side="right")

        container = tk.Frame(self, bg=c["BG"])
        container.pack(fill="both", expand=True, padx=12, pady=12)
        self._build_controls(container)

    def _build_controls(self, parent: tk.Frame):
        c = _COLORS

        def card(title: str) -> tk.Frame:
            tk.Label(parent, text=title, bg=c["BG"], fg=c["HL"], font=("Consolas", 9, "bold")).pack(anchor="w")
            f = tk.Frame(parent, bg=c["CARD"], padx=10, pady=8)
            f.pack(fill="x", pady=(2, 8))
            return f

        def file_row(frm: tk.Frame, label: str, var: tk.StringVar, filetypes: list):
            r = tk.Frame(frm, bg=c["CARD"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, width=12, anchor="w", bg=c["CARD"], fg=c["FG"]).pack(side="left")
            tk.Entry(r, textvariable=var, bg=c["INPUT"], fg=c["FG"],
                     insertbackground=c["FG"], relief="flat",
                     highlightthickness=0).pack(side="left", fill="x", expand=True)
            ttk.Button(r, text="..", width=3, command=lambda v=var, ft=filetypes: v.set(filedialog.askopenfilename(filetypes=ft))).pack(side="left", padx=2)

        def param_row(frm: tk.Frame, label: str, var, tooltip: str = ""):
            r = tk.Frame(frm, bg=c["CARD"])
            r.pack(fill="x", pady=1)
            tk.Label(r, text=label, width=12, anchor="w", bg=c["CARD"], fg=c["FG"]).pack(side="left")
            if isinstance(var, tk.IntVar):
                tk.Spinbox(r, textvariable=var, from_=0, to=9999, increment=1,
                           width=10,
                           bg=c["INPUT"], fg=c["FG"], insertbackground=c["FG"],
                           buttonbackground=c["ACC"], relief="flat",
                           highlightthickness=0).pack(side="left")
            elif isinstance(var, tk.DoubleVar):
                tk.Spinbox(r, textvariable=var, from_=0, to=9999, increment=0.001,
                           width=10,
                           bg=c["INPUT"], fg=c["FG"], insertbackground=c["FG"],
                           buttonbackground=c["ACC"], relief="flat",
                           highlightthickness=0).pack(side="left")
            else:
                tk.Entry(r, textvariable=var, width=10,
                         bg=c["INPUT"], fg=c["FG"], insertbackground=c["FG"],
                         relief="flat", highlightthickness=0).pack(side="left")
            if tooltip:
                tk.Label(r, text=f"  ⓘ {tooltip}", bg=c["CARD"], fg="#666", font=("Consolas", 8)).pack(side="left", padx=(4, 0))

        # ── FILES ──
        f1 = card("📁  FILES")
        self.yaml_var  = tk.StringVar()
        self.model_var = tk.StringVar(value="yolo26n.pt")
        self.checkpoint_var = tk.StringVar()
        file_row(f1, "Dataset", self.yaml_var, [("YAML", "*.yaml"), ("All", "*.*")])
        file_row(f1, "Model", self.model_var, [("Weights", "*.pt"), ("All", "*.*")])
        file_row(f1, "Checkpoint", self.checkpoint_var, [("Weights", "*.pt"), ("All", "*.*")])

        # ── PARAMETERS ──
        f2 = card("⚙️  PARAMETERS")
        self.ep_v   = tk.IntVar(value=EPOCHS)
        self.im_v   = tk.IntVar(value=IMGSZ)
        self.ba_v   = tk.IntVar(value=BATCH)
        self.wo_v   = tk.IntVar(value=4)
        self.lr0_v  = tk.DoubleVar(value=0.001)
        self.lrf_v  = tk.DoubleVar(value=0.01)
        self.wa_v   = tk.IntVar(value=5)
        self.conf_v = tk.DoubleVar(value=0.25)
        self.iou_v  = tk.DoubleVar(value=0.45)
        self.cache_v = tk.StringVar(value="disk")

        param_row(f2, "Epochs",  self.ep_v)
        param_row(f2, "ImgSz",   self.im_v)
        param_row(f2, "Batch",   self.ba_v)
        param_row(f2, "Workers", self.wo_v)
        param_row(f2, "lr0",     self.lr0_v)
        param_row(f2, "lrf",     self.lrf_v)
        param_row(f2, "Warmup",  self.wa_v)
        param_row(f2, "Conf",    self.conf_v)
        param_row(f2, "IOU",     self.iou_v, tooltip="NMS overlap threshold (default 0.45).")

        # ── Styled Cache row ──
        r_cache = tk.Frame(f2, bg=c["CARD"])
        r_cache.pack(fill="x", pady=1)
        tk.Label(r_cache, text="Cache", width=12, anchor="w", bg=c["CARD"], fg=c["FG"]).pack(side="left")
        om = tk.OptionMenu(r_cache, self.cache_v, "none", "ram", "disk")
        om.config(bg=c["INPUT"], fg=c["FG"], activebackground=c["HL"],
                  activeforeground=c["FG"], highlightthickness=0,
                  relief="flat", font=("Consolas", 10), width=7,
                  indicatoron=True, bd=0)
        om["menu"].config(bg=c["INPUT"], fg=c["FG"],
                          activebackground=c["HL"], activeforeground=c["FG"],
                          font=("Consolas", 10), bd=0)
        om.pack(side="left")

        # ── LOSS WEIGHTS ──
        f_loss = card("⚖️  LOSS WEIGHTS")
        self.cls_v = tk.DoubleVar(value=0.5)
        self.box_v = tk.DoubleVar(value=7.5)
        param_row(f_loss, "cls", self.cls_v, tooltip="Class loss gain (default 0.5).")
        param_row(f_loss, "box", self.box_v, tooltip="Box loss gain (default 7.5).")

        # ── DEVICE ──
        f3 = card("💻  DEVICE")
        self.dev_v = tk.StringVar(value=DEVICE)
        for opt in ("cpu", "cuda", "mps"):
            tk.Radiobutton(f3, text=opt.upper(), variable=self.dev_v, value=opt, bg=c["CARD"], fg=c["FG"], selectcolor=c["ACC"], activebackground=c["CARD"], font=("Consolas", 10)).pack(side="left", padx=5)

        # ── OPTIONS ──
        f4 = card("🔁  OPTIONS")
        self.show_labels_v = tk.BooleanVar(value=True)
        tk.Checkbutton(f4, text="Show labels on test image", variable=self.show_labels_v,
                       bg=c["CARD"], fg=c["FG"], selectcolor=c["ACC"],
                       activebackground=c["CARD"], activeforeground=c["HL"],
                       font=("Consolas", 10), bd=0, highlightthickness=0).pack(anchor="w")

        # ── Buttons ──
        r1 = tk.Frame(parent, bg=c["BG"])
        r1.pack(fill="x", pady=(2, 2))
        self.train_btn = ttk.Button(r1, text="▶  START TRAINING", command=self.start_training)
        self.resume_btn = ttk.Button(r1, text="↻  RESUME", style="Resume.TButton", command=self.resume_training)
        self.stop_btn  = ttk.Button(r1, text="■  STOP", style="Stop.TButton", command=self.stop_training, state="disabled")
        self.test_btn  = ttk.Button(r1, text="🔍  TEST IMAGE", command=self.run_test_image)
        self.train_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.resume_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.stop_btn.pack(side="left",  expand=True, fill="x", padx=(0, 2))
        self.test_btn.pack(side="left",  expand=True, fill="x")

        r2 = tk.Frame(parent, bg=c["BG"])
        r2.pack(fill="x", pady=(0, 4))
        self.fix_btn   = ttk.Button(r2, text="🔧  FIX DATASET", style="Fix.TButton", command=self.fix_dataset)
        self.clean_btn = ttk.Button(r2, text="🧹  CLEAN DATASET", command=self.open_clean_dialog)
        self.os_btn    = ttk.Button(r2, text="⚖  OVERSAMPLE", style="OS.TButton", command=self.oversample_dataset)
        self.fix_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.clean_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.os_btn.pack(side="left", expand=True, fill="x")


if __name__ == "__main__":
    app = TrainerApp()
    app.mainloop()