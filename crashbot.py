# full_advanced_crash_bot.py
import os
import datetime
import threading
import time
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog
import numpy as np
import pandas as pd
import mss
import cv2
import easyocr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
import joblib

# ---------- USER SETTINGS ----------
DATA_CSV = "crash_rounds_data.csv"    # persistent dataset
MODEL_CLASS_FILE = "clf_model.joblib"
MODEL_REG_FILE = "reg_model.joblib"
DEFAULT_TARGET_MULTIPLIER = 2.0
HISTORY_WINDOW = 600           # how many recent rounds kept in memory for features
FEATURE_ROUNDS = 20             # how many lags to use for feature vector
MIN_DATA_TO_TRAIN = 150         # require at least this many examples to train
POLL_INTERVAL = 0.5
SMOOTH_THRESHOLD = 0.05
AUTO_TRAIN = True
AUTO_TRAIN_INTERVAL = 600       # seconds (10 min)

# ---------- Helper: feature engineering ----------
def extract_features_from_history(
    history, target=DEFAULT_TARGET_MULTIPLIER, n_lags=FEATURE_ROUNDS,
    alpha=0.3, weight_type="linear"
):
    if len(history) < n_lags:
        return None

    arr = np.array(list(history)[-n_lags:])
    feats = {}

    # --- lag features ---
    for i in range(n_lags):
        feats[f"lag_{i+1}"] = arr[-(i+1)]

    # deltas
    deltas = np.diff(arr)
    feats["delta_mean"] = float(np.mean(deltas))
    feats["delta_std"] = float(np.std(deltas))

    # simple stats
    feats["arr_mean"] = float(np.mean(arr))
    feats["arr_std"] = float(np.std(arr))
    feats["arr_min"] = float(np.min(arr))
    feats["arr_max"] = float(np.max(arr))

    # --- weighted EMA ---
    ema = arr[0]
    for val in arr[1:]:
        ema = alpha * val + (1 - alpha) * ema
    feats["ema"] = float(ema)

    # --- weighted mean/std ---
    if weight_type == "linear":
        weights = np.linspace(1, 2, n_lags)
    elif weight_type == "exponential":
        weights = 2 ** np.linspace(0, n_lags-1, n_lags)  # recent rounds much higher weight
    else:
        weights = np.ones(n_lags)

    feats["weighted_mean"] = float(np.average(arr, weights=weights))
    feats["weighted_std"] = float(np.sqrt(np.average((arr - feats["weighted_mean"])**2, weights=weights)))

    # percentiles
    for p in [10, 25, 50, 75, 90]:
        feats[f"p{p}"] = float(np.percentile(arr, p))

    # count above target, last value relative to target
    feats["count_ge_target"] = int(np.sum(arr >= target))
    feats["last_over_target"] = int(arr[-1] >= target)

    # streaks
    streak_above = streak_below = 0
    for val in arr[::-1]:
        if val >= target and streak_below == 0:
            streak_above += 1
        elif val < target and streak_above == 0:
            streak_below += 1
        else:
            break
    feats["streak_above"] = streak_above
    feats["streak_below"] = streak_below

    # coefficient of variation
    feats["cv"] = float(np.std(arr) / (np.mean(arr) + 1e-8))

    # slope
    x = np.arange(len(arr))
    feats["slope"] = float(np.polyfit(x, arr, 1)[0]) if np.std(x) > 0 else 0.0

    return feats


# ---------- Data manager: save / load rounds ----------
def append_round_to_csv(multiplier, target=DEFAULT_TARGET_MULTIPLIER, csv_path=DATA_CSV):
    """Append a round entry (multiplier, timestamp) to CSV for later training"""
    df = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "multiplier": float(multiplier),
        "target": float(target)
    }])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

# ----------load_dataset_from_csv with this ----------
def load_dataset_from_csv(csv_path=DATA_CSV, n_lags=FEATURE_ROUNDS):
    """Load CSV file and convert to X,y for classification/regression using rolling windows"""
    if not os.path.exists(csv_path):
        return None, None, None, None  # no data
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None, None, None
    multipliers = df['multiplier'].astype(float).values
    X_rows = []
    y_class = []
    y_reg = []
    feat_names = None
    for i in range(n_lags, len(multipliers)):
        history_window = multipliers[i-n_lags:i]
        target = df['target'].iloc[i]
        feats = extract_features_from_history(history_window, target=target, n_lags=n_lags)
        if feats is None:
            continue
        if feat_names is None:
            feat_names = list(feats.keys())
        X_rows.append(list(feats.values()))
        y_class.append(1 if multipliers[i] >= target else 0)
        y_reg.append(multipliers[i])
    if len(X_rows) == 0:
        return None, None, None, None
    X = np.array(X_rows)
    return X, np.array(y_class), np.array(y_reg), feat_names


# ---------- Advanced Predictor (new) ----------
class AdvancedPredictor:
    def __init__(self, target=DEFAULT_TARGET_MULTIPLIER, n_lags=FEATURE_ROUNDS,
                 alpha=0.3, weight_type="linear"):
        self.target = target
        self.n_lags = n_lags
        self.alpha = alpha
        self.weight_type = weight_type
        self.history = deque(maxlen=HISTORY_WINDOW)
        self.clf_pipeline = None
        self.reg_pipeline = None
        self.feature_names = None
        self.new_rounds_since_train = 0

    def make_feature_vector(self):
        history_values = [v for ts, v in self.history]
        feats = extract_features_from_history(
            history_values,
            target=self.target,
            n_lags=self.n_lags,
            alpha=self.alpha,
            weight_type=self.weight_type
        )
        if feats is None:
            return None, None
        return np.array(list(feats.values())).reshape(1, -1), list(feats.keys())

    def add_round(self, multiplier):
        ts = datetime.datetime.now()
        self.history.append((ts, float(multiplier)))
        append_round_to_csv(multiplier, target=self.target)
        self.new_rounds_since_train += 1
    
    def auto_train_if_needed(self, min_new=50, csv_path=DATA_CSV):
        """
        Retrain models automatically if at least min_new rounds added since last training
        """
        if self.new_rounds_since_train >= min_new:
            try:
                stats = self.train_models_from_csv(csv_path=csv_path)
                self.save_models()
                self.new_rounds_since_train = 0  # reset counter
                print(f"[AutoTrain] Model retrained on {stats['n_samples']} samples. Accuracy={stats['accuracy']:.3f}")
            except Exception as e:
                print("[AutoTrain] Error:", e)


    def predict_next_multiplier(self):
        vec, names = self.make_feature_vector()
        if vec is None or self.reg_pipeline is None:
            return None
        pred = self.reg_pipeline.predict(vec)[0]
        return float(pred)

    def predict_next_safety(self):
        vec, names = self.make_feature_vector()
        if vec is None or self.clf_pipeline is None:
            return None
        prob = self.clf_pipeline.predict_proba(vec)[0][1]
        return float(prob)

    def risk_label(self, threshold_low=0.45, threshold_high=0.75):
        p = self.predict_next_safety()
        if p is None:
            return "Not enough data"
        if p >= threshold_high:
            return "Low Risk"
        elif p >= threshold_low:
            return "Medium Risk"
        else:
            return "High Risk"



    def train_models_from_csv(self, csv_path=DATA_CSV, min_rows=MIN_DATA_TO_TRAIN):
        # load and prepare dataset
        res = load_dataset_from_csv(csv_path, n_lags=self.n_lags)
        if res is None or res[0] is None:
            raise RuntimeError("Not enough data to build dataset.")
        X, y_class, y_reg, feat_names = res
        if X.shape[0] < min_rows:
            raise RuntimeError(f"Need at least {min_rows} rows, have {X.shape[0]}.")
        self.feature_names = feat_names
        # train-test split
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class)
        # classification pipeline
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        clf_pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        clf_pipe.fit(X_train, y_class_train)
        # regression pipeline
        reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        reg_pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])
        reg_pipe.fit(X_train, y_reg_train)
        # store
        self.clf_pipeline = clf_pipe
        self.reg_pipeline = reg_pipe
        # evaluation
        y_class_pred = clf_pipe.predict(X_test)
        y_class_prob = clf_pipe.predict_proba(X_test)[:,1]
        y_reg_pred = reg_pipe.predict(X_test)
        acc = accuracy_score(y_class_test, y_class_pred)
        ll = log_loss(y_class_test, np.vstack([1-y_class_prob, y_class_prob]).T)
        mse = mean_squared_error(y_reg_test, y_reg_pred)
        return {"accuracy": acc, "log_loss": ll, "mse": mse, "n_samples": X.shape[0]}

    def save_models(self, clf_path=MODEL_CLASS_FILE, reg_path=MODEL_REG_FILE):
        if self.clf_pipeline is not None:
            joblib.dump(self.clf_pipeline, clf_path)
        if self.reg_pipeline is not None:
            joblib.dump(self.reg_pipeline, reg_path)

    def load_models(self, clf_path=MODEL_CLASS_FILE, reg_path=MODEL_REG_FILE):
        if os.path.exists(clf_path):
            self.clf_pipeline = joblib.load(clf_path)
        if os.path.exists(reg_path):
            self.reg_pipeline = joblib.load(reg_path)

# ---------- OCR (unchanged, integrated) ----------
class SmoothOCRReader:
    def __init__(self, buffer_size=6, min_value=1.0, max_value=100.0, gpu=False):
        self.buffer = deque(maxlen=buffer_size)  # for smoothing/display only
        self.min_value = min_value
        self.max_value = max_value
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.detect_color_ranges = []  # list of acceptable color ranges
        self.last_raw_value = None      # store last OCR read

    def add_color_range_from_rgb(self, rgb):
        rgb_np = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)[0][0]
        h = int(hsv[0])
        lower = np.array([max(0, h-12), 60, 60])
        upper = np.array([min(179, h+12), 255, 255])
        self.detect_color_ranges.append((lower, upper))

    def process_frame(self, img_bgr):
        if img_bgr is None:
            return None, None
        try:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        except:
            return None, None

        # Mask by color ranges
        if self.detect_color_ranges:
            combined_mask = None
            for lower, upper in self.detect_color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            if cv2.countNonZero(combined_mask) == 0:
                return None, None
            img_masked = cv2.bitwise_and(img_bgr, img_bgr, mask=combined_mask)
        else:
            img_masked = img_bgr
            combined_mask = np.ones(img_bgr.shape[:2], dtype=np.uint8)*255

        gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(combined_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            gray = gray[y1:y2+1, x1:x2+1]

        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb_img = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)

        try:
            results = self.reader.readtext(rgb_img, allowlist='0123456789.')
        except:
            return None, None

        candidate = None
        for res in results:
            text = res[1].strip().replace(',', '.').replace(' ', '')
            filtered = ''.join(ch for ch in text if (ch.isdigit() or ch == '.'))
            if filtered == '' or filtered == '.':
                continue
            if filtered.count('.') > 1:
                continue
            try:
                val = float(filtered)
                if self.min_value <= val <= self.max_value:
                    candidate = val
                    break
            except:
                continue

        if candidate is None:
            return None, None

        # store last raw value
        self.last_raw_value = candidate

        # append to smoothing buffer
        self.buffer.append(candidate)
        smoothed = float(np.median(list(self.buffer)))

        # return both: raw and smoothed
        return candidate, smoothed


# ---------- GUI with advanced controls ----------
class CrashBotGUI:
    def __init__(self, root):
        self.root = root
        root.title("Advanced Crash ML Bot")
        self.predictor = AdvancedPredictor(target=DEFAULT_TARGET_MULTIPLIER, n_lags=FEATURE_ROUNDS)
        self.ocr = SmoothOCRReader(gpu=False)
        self.region = None
        self.ocr_running = False
        self.last_tracked_value = None
        self.current_reading = None
        self.auto_train_enabled = AUTO_TRAIN
        self._build_ui()

        # attempt to load models if exist
        self.predictor.load_models()

        # auto-train thread (optional)
        if self.auto_train_enabled:
            threading.Thread(target=self._auto_train_loop, daemon=True).start()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8); frm.pack(fill=tk.BOTH, expand=True)

        # Top controls
        top = ttk.Frame(frm); top.pack(fill=tk.X, pady=4)
        ttk.Button(top, text="Select Region", command=self.select_region).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Pick Color", command=self.pick_color).pack(side=tk.LEFT, padx=3)
        self.start_btn = ttk.Button(top, text="Start OCR", command=self.toggle_ocr, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Train Models", command=self.train_models).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Models", command=self.load_models).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save Models", command=self.save_models).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=3)

        # Prediction display
        pred_frame = ttk.LabelFrame(frm, text="Prediction")
        pred_frame.pack(fill=tk.X, pady=6)
        # ---------- Risk Threshold sliders ----------
        risk_frame = ttk.LabelFrame(frm, text="Risk Thresholds")
        risk_frame.pack(fill=tk.X, pady=6)

        # Low Risk Threshold
        ttk.Label(risk_frame, text="Low Risk ≥").pack(side=tk.LEFT, padx=2)
        self.low_thresh_var = tk.DoubleVar(value=0.75)  # default 75%
        ttk.Scale(risk_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.low_thresh_var, length=150).pack(side=tk.LEFT, padx=2)
        ttk.Label(risk_frame, text="(75%)").pack(side=tk.LEFT, padx=2)

        # Medium Risk Threshold
        ttk.Label(risk_frame, text="Medium Risk ≥").pack(side=tk.LEFT, padx=10)
        self.med_thresh_var = tk.DoubleVar(value=0.45)  # default 45%
        ttk.Scale(risk_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.med_thresh_var, length=150).pack(side=tk.LEFT, padx=2)
        ttk.Label(risk_frame, text="(45%)").pack(side=tk.LEFT, padx=2)

        # --- Auto-Train Settings ---
        auto_frame = ttk.LabelFrame(frm, text="Auto-Train Settings")
        auto_frame.pack(fill=tk.X, pady=6)

        ttk.Label(auto_frame, text="Min new rounds to auto-train:").pack(side=tk.LEFT, padx=2)
        self.min_new_var = tk.IntVar(value=50)
        ttk.Spinbox(auto_frame, from_=1, to=500, textvariable=self.min_new_var, width=5).pack(side=tk.LEFT, padx=2)


        ttk.Label(pred_frame, text="Last Tracked:").grid(row=0, column=0, sticky=tk.W)
        self.last_tracked_var = tk.StringVar(value="N/A"); ttk.Label(pred_frame, textvariable=self.last_tracked_var).grid(row=0,column=1,sticky=tk.W)
        ttk.Label(pred_frame, text="Predicted Next:").grid(row=1, column=0, sticky=tk.W)
        self.pred_next_var = tk.StringVar(value="N/A"); ttk.Label(pred_frame, textvariable=self.pred_next_var).grid(row=1,column=1,sticky=tk.W)
        ttk.Label(pred_frame, text="Prob >= target:").grid(row=2, column=0, sticky=tk.W)
        self.prob_var = tk.StringVar(value="N/A"); ttk.Label(pred_frame, textvariable=self.prob_var).grid(row=2,column=1,sticky=tk.W)
        ttk.Label(pred_frame, text="Risk:").grid(row=3,column=0,sticky=tk.W)
        self.risk_var = tk.StringVar(value="N/A"); ttk.Label(pred_frame, textvariable=self.risk_var).grid(row=3,column=1,sticky=tk.W)

        # --- Round Tracker Labels ---
        ttk.Label(pred_frame, text="Total Rounds Tracked:").grid(row=4, column=0, sticky=tk.W)
        self.total_rounds_var = tk.StringVar(value="0")
        ttk.Label(pred_frame, textvariable=self.total_rounds_var).grid(row=4, column=1, sticky=tk.W)

        ttk.Label(pred_frame, text="Rounds Since Last Train:").grid(row=5, column=0, sticky=tk.W)
        self.rounds_since_train_var = tk.StringVar(value="0")
        ttk.Label(pred_frame, textvariable=self.rounds_since_train_var).grid(row=5, column=1, sticky=tk.W)

        # Lists for history/current readings
        lists = ttk.Frame(frm); lists.pack(fill=tk.BOTH, expand=True)
        # tracked (left)
        tracked_frame = ttk.LabelFrame(lists, text="Rounds Tracked (unique changes)")
        tracked_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.tracked_listbox = tk.Listbox(tracked_frame, height=12); self.tracked_listbox.pack(fill=tk.BOTH, expand=True)
        # ---------- in _build_ui(), after tracked_listbox ----------
        # Recent rounds buttons
        self.recent_btn_frame = ttk.Frame(tracked_frame)
        self.recent_btn_frame.pack(fill=tk.X, pady=2)
        self.recent_buttons = []
        for i in range(5):
            btn = ttk.Button(self.recent_btn_frame, text="N/A", width=8,
                            command=lambda idx=i: self.remove_recent(idx))
            btn.pack(side=tk.LEFT, padx=2)
            self.recent_buttons.append(btn)

        # current readings (right)
        curr_frame = ttk.LabelFrame(lists, text="Currently Reading (OCR stream)")
        curr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.curr_listbox = tk.Listbox(curr_frame, height=12); self.curr_listbox.pack(fill=tk.BOTH, expand=True)

        # Manual input and controls bottom
        bottom = ttk.Frame(frm); bottom.pack(fill=tk.X, pady=4)
        ttk.Label(bottom, text="Manual add:").pack(side=tk.LEFT, padx=2)
        self.manual_var = tk.StringVar()
        ttk.Entry(bottom, textvariable=self.manual_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="Add", command=self.add_manual).pack(side=tk.LEFT, padx=2)
        ttk.Label(bottom, text="Target x:").pack(side=tk.LEFT, padx=8)
        self.target_var = tk.DoubleVar(value=DEFAULT_TARGET_MULTIPLIER)
        ttk.Entry(bottom, textvariable=self.target_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="Set Target", command=self.set_target).pack(side=tk.LEFT, padx=6)
        # Color ranges frame
        color_frame = ttk.LabelFrame(frm, text="OCR Color Ranges")
        color_frame.pack(fill=tk.X, pady=6)

        self.color_listbox = tk.Listbox(color_frame, height=5)
        self.color_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)

        # Remove button
        ttk.Button(color_frame, text="Remove Selected Color", command=self.remove_selected_color).pack(side=tk.LEFT, padx=4)


    # ---------- UI actions ----------
    def select_region(self):
        selector = RegionSelector(self.root); self.root.wait_window(selector)
        if selector.selection:
            self.region = selector.selection
            self.start_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Region set", f"Region set: {self.region}")

    def pick_color(self):
        c = colorchooser.askcolor(title="Pick OCR number color")
        if c and c[0]:
            rgb = tuple(int(x) for x in c[0])
            self.ocr.add_color_range_from_rgb(rgb)
            # refresh the listbox entirely to keep indices consistent
            self._refresh_color_listbox()
            messagebox.showinfo("Color added", f"Added color RGB: {rgb}\nTotal ranges: {len(self.ocr.detect_color_ranges)}")

    def remove_selected_color(self):
        sel = self.color_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.ocr.detect_color_ranges):
            removed = self.ocr.detect_color_ranges.pop(idx)
            self._refresh_color_listbox()
            messagebox.showinfo("Removed", f"Removed color range {removed}")
    def _refresh_color_listbox(self):
        # repopulate color_listbox; call from main thread only
        self.color_listbox.delete(0, tk.END)
        for i, (lower, upper) in enumerate(self.ocr.detect_color_ranges):
            self.color_listbox.insert(tk.END, f"{i}: HSV L{tuple(int(x) for x in lower)} U{tuple(int(x) for x in upper)}")




    def toggle_ocr(self):
        if not self.ocr_running:
            if not self.region:
                messagebox.showerror("No region", "Pick a region first")
                return
            self.ocr_running = True
            self.start_btn.config(text="Stop OCR")
            threading.Thread(target=self._ocr_loop, daemon=True).start()
        else:
            self.ocr_running = False
            self.start_btn.config(text="Start OCR")

    def add_manual(self):
        try:
            v = float(self.manual_var.get())
        except:
            messagebox.showerror("Invalid", "Enter numeric multiplier")
            return
        self._track_value(v)
        self.manual_var.set("")

    def set_target(self):
        t = float(self.target_var.get())
        self.predictor.target = t
        messagebox.showinfo("Target updated", f"Target set to {t}")

    def _track_value(self, v):
        """
        Record a value only when it changes significantly from the last tracked value.
        Uses SMOOTH_THRESHOLD to ignore tiny fluctuations.
        Only the last smoothed OCR reading per loop iteration is recorded.
        """
        last_val = self.last_tracked_value[1] if self.last_tracked_value else None

        # Only track if value is meaningfully different
        if last_val is None or abs(v - last_val) > SMOOTH_THRESHOLD:
            # Add to predictor/history
            self.predictor.add_round(v)

            # Update last tracked
            self.last_tracked_value = (datetime.datetime.now(), v)

            # Update GUI safely in main thread
            self.root.after(0, lambda v=v: self._insert_tracked((datetime.datetime.now(), v)))
            self.update_prediction_display()
            self.update_recent_buttons()
            # update visual round counters
            self.total_rounds_var.set(str(len(self.predictor.history)))
            self.rounds_since_train_var.set(str(self.predictor.new_rounds_since_train))




    # ---------- update_prediction_display ----------
    def update_prediction_display(self):
        if not self.predictor.history or not self.predictor.clf_pipeline or not self.predictor.reg_pipeline:
            self.pred_next_var.set("N/A")
            self.prob_var.set("N/A")
            self.risk_var.set("Not enough data")
            return

        vec, names = self.predictor.make_feature_vector()
        if vec is None:
            self.pred_next_var.set("N/A")
            self.prob_var.set("N/A")
            self.risk_var.set("Not enough data")
            return

        # regression & classification
        pred = self.predictor.reg_pipeline.predict(vec)[0]
        prob = self.predictor.clf_pipeline.predict_proba(vec)[0][1]

        self.pred_next_var.set(f"{pred:.2f}x")
        self.prob_var.set(f"{prob*100:.1f}%")
        risk = self.predictor.risk_label(
        threshold_low=self.med_thresh_var.get(),
        threshold_high=self.low_thresh_var.get()
        )
        self.risk_var.set(risk)




    # ---------- _ocr_loop with safe, non-blocking version ----------
    def _ocr_loop(self):
        with mss.mss() as sct:
            while self.ocr_running:
                try:
                    # Grab screen region; region must be set
                    if not self.region:
                        time.sleep(POLL_INTERVAL)
                        continue
                    frame = np.array(sct.grab(self.region))[:, :, :3]

                    # OCR returns (raw, smoothed)
                    raw_val, smoothed_val = self.ocr.process_frame(frame)

                    # Always append to current readings (for display/debugging) using smoothed
                    if smoothed_val is not None:
                        # push to main thread
                        self.root.after(0, lambda val=smoothed_val: self._insert_current(val))

                    # Track only raw value for history/predictions
                    if raw_val is not None:
                        self._track_value(raw_val)

                        # Auto-train using GUI-defined threshold
                        min_new_gui = self.min_new_var.get()  # fetch current Spinbox value
                        self.predictor.auto_train_if_needed(min_new=min_new_gui)

                    time.sleep(POLL_INTERVAL)
                except Exception as e:
                    print("OCR loop error:", e)
                    time.sleep(POLL_INTERVAL)
    
    def _auto_train_loop(self):
        while True:
            time.sleep(AUTO_TRAIN_INTERVAL)  # e.g., 5 min
            min_new_gui = self.min_new_var.get()  # fetch GUI value dynamically
            self.predictor.auto_train_if_needed(min_new=min_new_gui)

    # ---------- Recent buttons update ----------
    def update_recent_buttons(self):
        last5 = list(self.predictor.history)[-5:]
        for i in range(5):
            if i < len(last5):
                ts, val = last5[-(i+1)]
                self.recent_buttons[i].config(text=f"{val:.2f}x", state=tk.NORMAL)
            else:
                self.recent_buttons[i].config(text="N/A", state=tk.DISABLED)



    # ----------  remove_recent (fix CSV rewrite) ----------
    def remove_recent(self, idx):
        last5 = list(self.predictor.history)
        if idx < len(last5):
            removed = last5.pop(-(idx+1))  # match recent button ordering
            # rebuild deque
            self.predictor.history = deque(last5, maxlen=HISTORY_WINDOW)
            # rewrite CSV
            multipliers = [v for ts,v in self.predictor.history]
            df = pd.DataFrame({
                "timestamp": [ts.isoformat() for ts,_ in self.predictor.history],
                "multiplier": multipliers,
                "target": [self.predictor.target]*len(multipliers)
            })
            df.to_csv(DATA_CSV, index=False)
            self.update_recent_buttons()
            self.update_prediction_display()
            messagebox.showinfo("Removed", f"Removed value {removed[1]:.2f}x")




    def _insert_current(self, val):
        ts = datetime.datetime.now()
        self.curr_listbox.insert(tk.END, f"{ts.strftime('%H:%M:%S')} {val:.2f}x")
        self.curr_listbox.yview_moveto(1.0)



    # ---------- Training / model ops ----------
    def train_models(self):
        def _train():
            try:
                stats = self.predictor.train_models_from_csv()
            except Exception as e:
                messagebox.showerror("Training error", str(e))
                return
            msg = f"Training complete.\nAccuracy: {stats['accuracy']:.3f}\nLog loss: {stats['log_loss']:.3f}\nReg MSE: {stats['mse']:.4f}\nSamples: {stats['n_samples']}"
            messagebox.showinfo("Training finished", msg)
        threading.Thread(target=_train, daemon=True).start()

    def save_models(self):
        self.predictor.save_models()
        messagebox.showinfo("Saved", "Models saved to disk")

    def load_models(self):
        self.predictor.load_models()
        messagebox.showinfo("Loaded", "Models loaded (if present)")

    def export_csv(self):
        if not os.path.exists(DATA_CSV):
            messagebox.showerror("No data", "No CSV found")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if dest:
            import shutil
            shutil.copy(DATA_CSV, dest)
            messagebox.showinfo("Exported", f"Exported to {dest}")

    def _insert_tracked(self, v_tuple):
        ts, val = v_tuple
        # Only keep last 3 in the listbox
        self.tracked_listbox.delete(0, tk.END)
        last3 = list(self.predictor.history)[-3:]
        for ts2, val2 in last3:
            # Use spacing to separate time and multiplier
            line = f"{ts2.strftime('%H:%M:%S')}    {val2:.2f}x"
            self.tracked_listbox.insert(tk.END, line)
        self.tracked_listbox.yview_moveto(1.0)
        # Larger, bold font
        self.tracked_listbox.config(font=("TkDefaultFont", 16, "bold"))

# ---------- Region selector (same as before) ----------
class RegionSelector(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Select Region")
        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.4)
        self.configure(background='black')
        self.canvas = tk.Canvas(self, cursor="cross", bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start_x = self.start_y = None
        self.rect = None
        self.selection = None
        self.bind("<Escape>", lambda e: self.cancel())
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
    def on_press(self, e):
        self.start_x = self.canvas.canvasx(e.x); self.start_y = self.canvas.canvasy(e.y)
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
    def on_move(self, e):
        curX, curY = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
    def on_release(self, e):
        end_x, end_y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        left = int(min(self.start_x, end_x)); top = int(min(self.start_y, end_y))
        width = int(abs(end_x - self.start_x)); height = int(abs(end_y - self.start_y))
        if width < 5 or height < 5:
            messagebox.showerror("Selection too small", "Draw a larger rectangle.")
            return
        self.selection = {"left": left, "top": top, "width": width, "height": height}
        self.destroy()
    def cancel(self):
        self.selection = None; self.destroy()

# ---------- Main ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = CrashBotGUI(root)
    root.geometry("900x600")
    root.mainloop()
