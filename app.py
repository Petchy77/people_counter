import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
import json
import os
import subprocess
import re
from datetime import datetime
import numpy as np
import csv

if os.environ.get('DISPLAY') is None:
    os.environ['DISPLAY'] = ':0'

# ======== CONFIG ========
MODEL_PATH    = '/home/pi/mask_detection/mask_detection.hef'
SETTINGS_PATH = '/home/pi/people_counter/settings.json'
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
# ========================

# ── Colors (UI) ──
BG        = "#0a0e1a"
PANEL     = "#111827"
BORDER    = "#1f2d40"
CARD      = "#111827"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
TEXT_INV  = "#ffffff"

C_IN      = "#00e5a0"
C_OUT     = "#ff4d6d"
C_BLUE    = "#38bdf8"
C_YELLOW  = "#facc15"
C_IN_BG   = "#002d20"
C_OUT_BG  = "#2d0010"
C_BLUE_BG = "#0a1f2d"

# ── Colors (OpenCV RGB) ──
CV_ORANGE = (255, 140, 0)   # Focus Area
CV_GREEN  = (0, 229, 160)   # Person Box & IN
CV_RED    = (255, 77, 109)  # OUT
CV_YELLOW = (250, 204, 21)  # Line

# ── Runtime ──
latest_frame    = None
latest_dets     = []   
latest_dets_wh  = (640, 480)
lock            = threading.Lock()
running         = False
cam_thread      = None
current_cam     = None
current_fps     = 0
current_live    = 0

# ── Counters ──
count_in    = 0
count_out   = 0
last_reset  = datetime.now().date()

# ── Tracked people ──
tracked     = {}   
next_id     = 0

# ── Settings ──
settings = {
    'confidence': 0.5,
    'rotate':     None,
    'model_path': MODEL_PATH,
    'line_pct':   0.5,   
    'log_path':   '/home/pi/people_counter/log',
    'roi':        [0.1, 0.1, 0.9, 0.9],
    'display_mode': 'fullscreen'
}

def load_settings():
    global settings
    try:
        with open(SETTINGS_PATH) as f:
            s = json.load(f)
        settings.update(s)
    except Exception:
        pass

def save_settings():
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    try:
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f)
    except Exception:
        pass

def load_log():
    global count_in, count_out
    log_dir = settings.get('log_path', '/home/pi/people_counter/log')
    today_str = datetime.now().strftime("%Y-%m-%d")
    month_str = datetime.now().strftime("%Y-%m")
    filepath = os.path.join(log_dir, f"log_{month_str}.csv")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    if len(r) >= 3 and r[0] == today_str:
                        count_in = int(r[1])
                        count_out = int(r[2])
        except: pass

def save_log():
    log_dir = settings.get('log_path', '/home/pi/people_counter/log')
    os.makedirs(log_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    month_str = datetime.now().strftime("%Y-%m")
    filepath = os.path.join(log_dir, f"log_{month_str}.csv")
    
    rows = []
    found = False
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    if len(r) >= 3 and r[0] == today_str:
                        rows.append([today_str, count_in, count_out])
                        found = True
                    else:
                        rows.append(r)
        except: pass
        
    if not found:
        if not os.path.exists(filepath):
            rows.append(['Date', 'In', 'Out'])
        rows.append([today_str, count_in, count_out])
        
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except: pass

load_settings()
load_log()


# ─────────────────────────────────────────────────────────────────────────────
# Camera scan
# ─────────────────────────────────────────────────────────────────────────────
def scan_cameras():
    cameras = []
    try:
        from picamera2 import Picamera2
        infos = Picamera2.global_camera_info()
        seen_names = set()
        for i, c in enumerate(infos):
            model = c.get('Model', '')
            is_usb = 'uvcvideo' in c.get('Id','').lower() or 'usb' in c.get('Id','').lower()
            cam_type = 'usb' if is_usb else 'picam'
            name = f"Pi Camera ({model})" if not is_usb else f"USB Camera ({model})"
            if name not in seen_names:
                seen_names.add(name)
                cameras.append({'id': f'picam_{i}', 'name': name,
                                'type': cam_type, 'index': i})
    except Exception:
        pass
    try:
        out = subprocess.check_output(['v4l2-ctl','--list-devices'],
                                      stderr=subprocess.DEVNULL).decode()
        cur = ''
        seen = set()
        for line in out.splitlines():
            if not line.startswith('\t') and not line.startswith('/dev/video'):
                cur = line.strip()
                continue
            dev = line.strip()
            if not dev.startswith('/dev/video'):
                continue
            if 'rp1' in cur.lower() or 'cfe' in cur.lower() or 'isp' in cur.lower() or 'pispbe' in cur.lower() or 'hevc' in cur.lower():
                continue
            m = re.search(r'video(\d+)', dev)
            if not m:
                continue
            idx = int(m.group(1))
            if idx > 10:
                continue
            cam_name = cur.split("(")[0].strip() or f"video{idx}"
            if cam_name in seen:
                continue
            seen.add(cam_name)
            if not any(c['index'] == idx for c in cameras):
                cameras.append({'id': f'usb_{idx}',
                                'name': f'USB Camera ({cam_name})',
                                'type': 'usb', 'index': idx})
    except Exception:
        pass
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
# Camera + Inference loop
# ─────────────────────────────────────────────────────────────────────────────
def camera_loop(cam, on_count, screen_w, screen_h):
    def _show_error(msg):
        global latest_frame
        h, w = screen_h, screen_w
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 10, 10)
        y = h // 2 - 40
        for line in msg.split('\n'):
            cv2.putText(frame, line, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            y += 36
        with lock:
            latest_frame = frame

    global latest_frame, latest_dets, latest_dets_wh, running, count_in, count_out, tracked, next_id, current_fps, current_live

    from picamera2 import Picamera2
    picam = Picamera2(cam['index'])
    picam.configure(picam.create_preview_configuration(
        main={"format": "RGB888", "size": (1920, 1080)}))
    try:
        picam.set_controls({"AfMode": 2, "AfTrigger": 0})
    except Exception:
        pass
    picam.start()

    from hailo_platform import (HEF, VDevice, FormatType,
                                HailoSchedulingAlgorithm,
                                InferVStreams, InputVStreamParams,
                                OutputVStreamParams)

    model_path = settings['model_path']

    try:
        hef = HEF(model_path)
    except Exception as e:
        err_msg = f"❌ โหลด Model ไม่ได้\n{os.path.basename(model_path)}\n{e}"
        _show_error(err_msg)
        return
        
    _shape = hef.get_input_vstream_infos()[0].shape
    INPUT_SIZE = _shape[0]
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    target = VDevice(params=params)
    ng     = target.configure(hef)[0]
    ngp    = ng.create_params()
    ivp    = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    ovp    = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    input_name = hef.get_input_vstream_infos()[0].name

    def preprocess(frame):
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        
        sq = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        py = (INPUT_SIZE - nh) // 2
        px = (INPUT_SIZE - nw) // 2
        sq[py:py+nh, px:px+nw] = resized
        
        preprocess.scale = scale
        preprocess.px = px
        preprocess.py = py
        return np.expand_dims(sq, 0)

    preprocess.scale = 1.0
    preprocess.px = 0
    preprocess.py = 0

    def detect(outputs, orig_h, orig_w, conf_thresh):
        key = list(outputs.keys())[0]
        out = outputs[key][0]
        results = []
        person_dets = out[0]
        if len(person_dets) == 0:
            return results
            
        scale = preprocess.scale
        px = preprocess.px
        py = preprocess.py
        
        for det in person_dets:
            y1, x1, y2, x2, score = det
            
            if score < (conf_thresh - 0.1):
                continue
                
            x1_sq = x1 * INPUT_SIZE
            y1_sq = y1 * INPUT_SIZE
            x2_sq = x2 * INPUT_SIZE
            y2_sq = y2 * INPUT_SIZE
            
            x1 = int(np.clip((x1_sq - px) / scale, 0, orig_w))
            y1 = int(np.clip((y1_sq - py) / scale, 0, orig_h))
            x2 = int(np.clip((x2_sq - px) / scale, 0, orig_w))
            y2 = int(np.clip((y2_sq - py) / scale, 0, orig_h))
            
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2, float(score)))
        return results

    fps_c, fps_t = 0, time.time()
    detection_fps = 0

    with ng.activate(ngp):
        with InferVStreams(ng, ivp, ovp) as pipe:
            while running:
                frame = picam.capture_array()
                rotate = settings.get('rotate')
                if rotate is not None:
                    frame = cv2.rotate(frame, rotate)

                orig_h, orig_w = frame.shape[:2]
                
                rx1, ry1, rx2, ry2 = settings.get('roi', [0.1, 0.1, 0.9, 0.9])
                roi_px1, roi_py1 = int(rx1 * orig_w), int(ry1 * orig_h)
                roi_px2, roi_py2 = int(rx2 * orig_w), int(ry2 * orig_h)
                line_y = int(settings['line_pct'] * orig_h)

                inp  = preprocess(frame)
                try:
                    outs = pipe.infer({input_name: inp})
                    dets = detect(outs, orig_h, orig_w, settings['confidence'])
                except Exception as e:
                    _show_error(f"❌ Inference Error\n{str(e)[:60]}")
                    time.sleep(2)
                    continue

                curr_time = time.time()
                new_tracked = {}
                live_count = 0
                
                for (x1,y1,x2,y2,conf) in dets:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    if not (roi_px1 <= cx <= roi_px2 and roi_py1 <= cy <= roi_py2):
                        continue
                    live_count += 1
                    
                    best_id, best_d = None, 120
                    for tid, tr in tracked.items():
                        d = abs(tr['cx']-cx) + abs(tr['cy']-cy)
                        if d < best_d:
                            best_d, best_id = d, tid
                            
                    if best_id is None:
                        next_id += 1
                        best_id = next_id
                        tracked[best_id] = {'cx':cx,'cy':cy,'counted':False,'last_seen':curr_time,'zone':None}
                        
                    tr = tracked[best_id]
                    current_zone = "A" if cy < (line_y - 10) else ("B" if cy > (line_y + 10) else tr['zone'])
                    
                    if not tr['counted']:
                        if tr['zone'] == "A" and current_zone == "B":
                            count_in += 1
                            tr['counted'] = True
                            on_count()
                        elif tr['zone'] == "B" and current_zone == "A":
                            count_out += 1
                            tr['counted'] = True
                            on_count()
                            
                    counted = tr['counted']
                    if counted and abs(cy - line_y) > 50:
                        counted = False
                        
                    new_tracked[best_id] = {
                        'cx': cx, 'cy': cy, 
                        'counted': counted, 
                        'last_seen': curr_time,
                        'zone': current_zone
                    }

                for tid, tr in tracked.items():
                    if tid not in new_tracked:
                        if curr_time - tr.get('last_seen', curr_time) <= 1.5:
                            new_tracked[tid] = tr

                tracked = new_tracked
                current_live = live_count

                fps_c += 1
                if time.time()-fps_t >= 1.0:
                    detection_fps = fps_c
                    current_fps = fps_c
                    fps_c, fps_t = 0, time.time()

                with lock:
                    latest_dets[:] = dets
                    latest_dets_wh = (orig_w, orig_h)
                    latest_frame = frame.copy()

    with lock:
        latest_frame = None
    try: picam.stop(); picam.close()
    except: pass


# ─────────────────────────────────────────────────────────────────────────────
# Settings Dialog
# ─────────────────────────────────────────────────────────────────────────────
class SettingsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("ตั้งค่า")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.grab_set()
        self.geometry("420x460")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width()-420)//2
        y = parent.winfo_y() + (parent.winfo_height()-460)//2
        self.geometry(f"+{x}+{y}")
        self._build()

    def _build(self):
        pad = tk.Frame(self, bg=PANEL)
        pad.pack(fill='both', expand=True, padx=24, pady=20)

        tk.Label(pad, text="⚙  ตั้งค่า", font=("Helvetica",14,"bold"), bg=PANEL, fg=TEXT).pack(anchor='w', pady=(0,16))

        tk.Label(pad, text="ความมั่นใจขั้นต่ำ (Confidence)", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(anchor='w')
        conf_row = tk.Frame(pad, bg=PANEL)
        conf_row.pack(fill='x', pady=(4,12))
        self._conf_var = tk.DoubleVar(value=settings['confidence'])
        self._conf_lbl = tk.Label(conf_row, text=f"{settings['confidence']:.0%}", font=("Helvetica",11,"bold"), bg=PANEL, fg=C_BLUE, width=5)
        self._conf_lbl.pack(side='right')
        tk.Scale(conf_row, from_=0.1, to=0.95, resolution=0.05, orient='horizontal', variable=self._conf_var, bg=PANEL, fg=TEXT, highlightthickness=0, troughcolor=BORDER, activebackground=C_BLUE, command=lambda v: self._conf_lbl.config(text=f"{float(v):.0%}"), showvalue=False).pack(side='left', fill='x', expand=True)

        tk.Label(pad, text="หมุนภาพกล้อง", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(anchor='w')
        rotate_opts = [("ไม่หมุน", None), ("หมุน 90° ตามเข็ม", cv2.ROTATE_90_CLOCKWISE), ("หมุน 90° ทวนเข็ม", cv2.ROTATE_90_COUNTERCLOCKWISE), ("หมุน 180°", cv2.ROTATE_180)]
        self._rotate_map = {l:v for l,v in rotate_opts}
        self._rotate_var = tk.StringVar()
        cur = settings['rotate']
        for l,v in rotate_opts:
            if v == cur:
                self._rotate_var.set(l); break
        else:
            self._rotate_var.set("ไม่หมุน")
        ttk.Combobox(pad, textvariable=self._rotate_var, values=[l for l,_ in rotate_opts], state='readonly', font=("Helvetica",10)).pack(fill='x', pady=(4,12))

        tk.Label(pad, text="Model (.hef)", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(anchor='w')
        self._model_var = tk.StringVar(value=settings['model_path'])
        mrow = tk.Frame(pad, bg=PANEL)
        mrow.pack(fill='x', pady=(4,12))
        tk.Entry(mrow, textvariable=self._model_var, font=("Helvetica",9), fg=TEXT, bg=BG, insertbackground=TEXT, relief='solid', bd=1).pack(side='left', fill='x', expand=True, padx=(0,6))
        tk.Button(mrow, text="เลือก", font=("Helvetica",10), bg=C_BLUE, fg='#000', relief='flat', padx=8, pady=4, command=self._browse_model).pack(side='right')

        tk.Label(pad, text="โฟลเดอร์เก็บ Log (CSV)", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(anchor='w')
        self._log_var = tk.StringVar(value=settings.get('log_path', '/home/pi/people_counter/log'))
        lrow = tk.Frame(pad, bg=PANEL)
        lrow.pack(fill='x', pady=(4,16))
        tk.Entry(lrow, textvariable=self._log_var, font=("Helvetica",9), fg=TEXT, bg=BG, insertbackground=TEXT, relief='solid', bd=1).pack(side='left', fill='x', expand=True, padx=(0,6))
        tk.Button(lrow, text="เลือก", font=("Helvetica",10), bg=C_BLUE, fg='#000', relief='flat', padx=8, pady=4, command=self._browse_log).pack(side='right')

        tk.Label(pad, text="โหมดแสดงผลหน้าจอ", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(anchor='w', pady=(8,0))
        disp_opts = [("Full Screen (เต็มจอ)", "fullscreen"), ("Windowed (หน้าต่าง)", "windowed")]
        self._disp_map = {l:v for l,v in disp_opts}
        self._disp_var = tk.StringVar()
        cur_disp = settings.get('display_mode', 'fullscreen')
        for l,v in disp_opts:
            if v == cur_disp:
                self._disp_var.set(l); break
        ttk.Combobox(pad, textvariable=self._disp_var, values=[l for l,_ in disp_opts], state='readonly', font=("Helvetica",10)).pack(fill='x', pady=(4,12))

        # --- Cancel / Save button ---
        brow = tk.Frame(pad, bg=PANEL)
        brow.pack(fill='x', side='bottom')
        tk.Button(brow, text="ยกเลิก", font=("Helvetica",11), bg=BORDER, fg=TEXT, relief='flat', padx=16, pady=6, command=self.destroy).pack(side='right', padx=(8,0))
        tk.Button(brow, text="บันทึก", font=("Helvetica",11,"bold"), bg=C_BLUE, fg='#000', relief='flat', padx=16, pady=6, command=self._save).pack(side='right')

    def _browse_model(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(initialdir='/home/pi', title='เลือก Model (.hef)', filetypes=[('HEF files','*.hef'),('All','*.*')])
        if p: self._model_var.set(p)

    def _browse_log(self):
        from tkinter import filedialog
        p = filedialog.askdirectory(initialdir=settings.get('log_path', '/home/pi'), title='เลือกโฟลเดอร์เก็บ Log')
        if p: self._log_var.set(p)

    def _save(self):
        settings['confidence'] = round(self._conf_var.get(), 2)
        settings['rotate']     = self._rotate_map[self._rotate_var.get()]
        settings['model_path'] = self._model_var.get()
        settings['log_path']   = self._log_var.get()
        
        settings['display_mode'] = self._disp_map[self._disp_var.get()]
        if settings['display_mode'] == 'fullscreen':
            self.master.attributes('-fullscreen', True)
        else:
            self.master.attributes('-fullscreen', False)
            self.master.geometry("1024x600")
            self.master.minsize(800, 480)
            
        save_settings()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
class PeopleCounterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("People Counter")
        self.configure(bg=BG)

        if settings.get('display_mode') == 'windowed':
            self.attributes('-fullscreen', False)
            self.geometry("1024x600")
            self.minsize(800, 480)
        else:
            self.attributes('-fullscreen', True)
            
        self.bind('<Escape>', lambda e: self.quit_app())
        self.bind('r',        lambda e: self._do_scan())
        self.bind('<Control-s>', lambda e: self._open_settings())

        self._cameras    = []
        self._cam_var    = tk.StringVar()
        self._drag_state = None
        self._drag_start = (0,0)
        self._roi_start  = []
        
        self._line_pct   = settings.get('line_pct', 0.5)
        self._roi        = settings.get('roi', [0.1, 0.1, 0.9, 0.9])

        self._build_ui()
        self._update_clock()
        self._update_video()
        self._check_midnight_reset()
        self.after(300, lambda: self._do_scan(auto_start=True))

    def _build_ui(self):
        hdr = tk.Frame(self, bg=PANEL, height=48)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=PANEL)
        left.pack(side='left', padx=16, fill='y')
        self._dot = tk.Label(left, text="●", font=("Helvetica",12), bg=PANEL, fg=C_IN)
        self._dot.pack(side='left', padx=(0,8))
        tk.Label(left, text="PEOPLE COUNTER", font=("Courier",12,"bold"), bg=PANEL, fg=TEXT).pack(side='left')

        right = tk.Frame(hdr, bg=PANEL)
        right.pack(side='right', padx=16, fill='y')
        tk.Button(right, text="✕  ออก", font=("Helvetica",11,"bold"), bg="#ef4444", fg=TEXT_INV, relief='flat', padx=12, pady=4, command=self.quit_app).pack(side='right', pady=8)

        self.lbl_clock = tk.Label(right, text="--:--:--", font=("Courier",13,"bold"), bg='#1a2236', fg=TEXT, padx=12, pady=4)
        self.lbl_clock.pack(side='right', padx=8, pady=8)

        self.lbl_fps = tk.Label(right, text="0 FPS", font=("Courier",11), bg=C_IN_BG, fg=C_IN, padx=10, pady=4)
        self.lbl_fps.pack(side='right', pady=8)

        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True)

        sw = 260
        sb = tk.Frame(body, bg=PANEL, width=sw)
        sb.pack(side='right', fill='y')
        sb.pack_propagate(False)

        left_space = tk.Frame(body, bg=BG)
        left_space.pack(side='left', fill='both', expand=True)

        cam_bar = tk.Frame(left_space, bg=PANEL, height=52)
        cam_bar.pack(fill='x', side='top', pady=(0, 2))
        cam_bar.pack_propagate(False)

        tk.Label(cam_bar, text="📷 เลือกกล้อง:", font=("Helvetica",10), bg=PANEL, fg=TEXT_DIM).pack(side='left', padx=(16, 8), pady=12)
        self.combo = ttk.Combobox(cam_bar, textvariable=self._cam_var, state='disabled', font=("Helvetica",10), width=35)
        self.combo.pack(side='left', pady=12)
        self.combo.bind('<<ComboboxSelected>>', self._on_cam_selected)
        
        self.btn_scan = tk.Button(cam_bar, text="⟳ สแกน [R]", font=("Helvetica",10,"bold"), bg='#1d4ed8', fg=TEXT_INV, relief='flat', padx=10, cursor='hand2', command=self._do_scan)
        self.btn_scan.pack(side='left', padx=12, pady=10)
        
        self.lbl_cam_status = tk.Label(cam_bar, text="", font=("Helvetica",9), bg=PANEL, fg=TEXT_DIM)
        self.lbl_cam_status.pack(side='left', pady=12)

        self.video_frame = tk.Frame(left_space, bg='#000')
        self.video_frame.pack(fill='both', expand=True)

        self.video_label = tk.Label(self.video_frame, bg='#000', cursor='crosshair')
        self.video_label.pack(fill='both', expand=True)
        self.video_label.bind('<ButtonPress-1>',   self._mouse_down)
        self.video_label.bind('<B1-Motion>',       self._mouse_move)
        self.video_label.bind('<ButtonRelease-1>', self._mouse_up)
        self.video_label.bind('<Motion>',          self._mouse_hover)

        self._build_section(sb, "📊  สถิติวันนี้", self._build_stats_section)
        
        ctrl = tk.Frame(sb, bg=PANEL)
        ctrl.pack(fill='x', padx=14, pady=8)
        tk.Button(ctrl, text="⚙  ตั้งค่า [Ctrl+S]", font=("Helvetica",11), bg=BORDER, fg=TEXT, relief='flat', pady=8, cursor='hand2', command=self._open_settings).pack(fill='x')

    def _build_section(self, parent, title, builder):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill='x', padx=0, pady=0)
        tk.Frame(f, bg=BORDER, height=1).pack(fill='x')
        tk.Label(f, text=title, font=("Helvetica",10,"bold"), bg=PANEL, fg=TEXT_DIM, padx=14, pady=8).pack(anchor='w')
        builder(f)

    def _build_stats_section(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill='x', padx=14, pady=(0,10))

        badge = tk.Frame(f, bg=C_BLUE_BG, highlightbackground='#1a3a4a', highlightthickness=1)
        badge.pack(fill='x', pady=(0,10))
        self.lbl_reset = tk.Label(badge, text=f"🔄  Reset ทุกเที่ยงคืน · {self._today_str()}", font=("Helvetica",9), bg=C_BLUE_BG, fg=C_BLUE, padx=10, pady=6)
        self.lbl_reset.pack(anchor='w')

        grid = tk.Frame(f, bg=PANEL)
        grid.pack(fill='x', pady=(0,8))
        grid.columnconfigure(0, weight=1)

        card_in = tk.Frame(grid, bg=C_IN_BG, highlightbackground=C_IN, highlightthickness=1)
        card_in.grid(row=0, column=0, sticky='nsew', pady=(0,4))
        tk.Label(card_in, text="↓ เข้า", font=("Helvetica",14,"bold"), bg=C_IN_BG, fg=C_IN).grid(row=0, column=0, sticky='nw', padx=12, pady=12)
        self.lbl_in = tk.Label(card_in, text="0", font=("Courier",36,"bold"), bg=C_IN_BG, fg=C_IN)
        self.lbl_in.grid(row=0, column=1, rowspan=2, sticky='e', padx=(0,4), pady=12)
        tk.Label(card_in, text="คน", font=("Helvetica",9), bg=C_IN_BG, fg=TEXT_DIM).grid(row=1, column=2, sticky='sw', padx=(0,12), pady=(0,8))
        card_in.columnconfigure(1, weight=1)

        card_out = tk.Frame(grid, bg=C_OUT_BG, highlightbackground=C_OUT, highlightthickness=1)
        card_out.grid(row=1, column=0, sticky='nsew', pady=(4,0))
        tk.Label(card_out, text="↑ ออก", font=("Helvetica",14,"bold"), bg=C_OUT_BG, fg=C_OUT).grid(row=0, column=0, sticky='nw', padx=12, pady=12)
        self.lbl_out = tk.Label(card_out, text="0", font=("Courier",36,"bold"), bg=C_OUT_BG, fg=C_OUT)
        self.lbl_out.grid(row=0, column=1, rowspan=2, sticky='e', padx=(0,4), pady=12)
        tk.Label(card_out, text="คน", font=("Helvetica",9), bg=C_OUT_BG, fg=TEXT_DIM).grid(row=1, column=2, sticky='sw', padx=(0,12), pady=(0,8))
        card_out.columnconfigure(1, weight=1)

        # Live Count 
        net = tk.Frame(f, bg=C_BLUE_BG, highlightbackground='#1a3a4a', highlightthickness=1)
        net.pack(fill='x', pady=(0,6))
        tk.Label(net, text="🏠  คนในพื้นที่ตอนนี้", font=("Helvetica",10,"bold"), bg=C_BLUE_BG, fg=C_BLUE).pack(side='left', padx=12, pady=10)
        self.lbl_net = tk.Label(net, text="0", font=("Courier",22,"bold"), bg=C_BLUE_BG, fg=C_BLUE)
        self.lbl_net.pack(side='right', padx=12)

        up = tk.Frame(f, bg='#1a1a00', highlightbackground='#3a3a00', highlightthickness=1)
        up.pack(fill='x')
        tk.Label(up, text="⚡  เปิดใช้งาน", font=("Helvetica",10), bg='#1a1a00', fg=C_YELLOW).pack(side='left', padx=12, pady=8)
        self._start_time = time.time()
        self.lbl_uptime = tk.Label(up, text="00:00:00", font=("Courier",11,"bold"), bg='#1a1a00', fg=C_YELLOW)
        self.lbl_uptime.pack(side='right', padx=12)

    def _mouse_down(self, e):
        if not hasattr(self, '_video_ox'): return
        ox, oy = self._video_ox, self._video_oy
        nw, nh = self._video_nw, self._video_nh
        nx = max(0.0, min(1.0, (e.x - ox) / nw))
        ny = max(0.0, min(1.0, (e.y - oy) / nh))
        
        rx1, ry1, rx2, ry2 = self._roi

        if rx1 <= nx <= rx2 and abs(ny - self._line_pct) < 0.05:
            self._drag_state = 'line'
            return

        handles = [
            (rx1, ry1), ((rx1+rx2)/2, ry1), (rx2, ry1),
            (rx1, (ry1+ry2)/2),             (rx2, (ry1+ry2)/2),
            (rx1, ry2), ((rx1+rx2)/2, ry2), (rx2, ry2)
        ]
        for i, (hx, hy) in enumerate(handles):
            if abs(nx - hx) < 0.04 and abs(ny - hy) < 0.04:
                self._drag_state = f'roi_{i}'
                return
                
        if rx1 < nx < rx2 and ry1 < ny < ry2:
            self._drag_state = 'roi_8'
            self._drag_start = (nx, ny)
            self._roi_start = list(self._roi)

    def _mouse_move(self, e):
        if getattr(self, '_drag_state', None) is None: return
        if not hasattr(self, '_video_ox'): return
        ox, oy = self._video_ox, self._video_oy
        nw, nh = self._video_nw, self._video_nh
        nx = max(0.0, min(1.0, (e.x - ox) / nw))
        ny = max(0.0, min(1.0, (e.y - oy) / nh))

        if self._drag_state == 'line':
            settings['line_pct'] = self._line_pct = ny
        elif self._drag_state.startswith('roi_'):
            idx = int(self._drag_state.split('_')[1])
            rx1, ry1, rx2, ry2 = self._roi
            
            if idx == 0: rx1, ry1 = nx, ny
            elif idx == 1: ry1 = ny
            elif idx == 2: rx2, ry1 = nx, ny
            elif idx == 3: rx1 = nx
            elif idx == 4: rx2 = nx
            elif idx == 5: rx1, ry2 = nx, ny
            elif idx == 6: ry2 = ny
            elif idx == 7: rx2, ry2 = nx, ny
            elif idx == 8:
                dx = nx - self._drag_start[0]
                dy = ny - self._drag_start[1]
                w = self._roi_start[2] - self._roi_start[0]
                h = self._roi_start[3] - self._roi_start[1]
                rx1 = max(0.0, min(1.0 - w, self._roi_start[0] + dx))
                ry1 = max(0.0, min(1.0 - h, self._roi_start[1] + dy))
                rx2, ry2 = rx1 + w, ry1 + h

            if rx1 < rx2 and ry1 < ry2:
                settings['roi'] = self._roi = [rx1, ry1, rx2, ry2]

    def _mouse_up(self, e):
        self._drag_state = None
        save_settings()

    def _mouse_hover(self, e):
        if getattr(self, '_drag_state', None) is not None:
            return 
            
        if not hasattr(self, '_video_ox'): return
        ox, oy = self._video_ox, self._video_oy
        nw, nh = self._video_nw, self._video_nh
        if nw == 0 or nh == 0: return
        nx = max(0.0, min(1.0, (e.x - ox) / nw))
        ny = max(0.0, min(1.0, (e.y - oy) / nh))
        
        rx1, ry1, rx2, ry2 = self._roi
        
        if rx1 <= nx <= rx2 and abs(ny - self._line_pct) < 0.05:
            self.video_label.config(cursor='sb_v_double_arrow')
            return
            
        handles = [
            (rx1, ry1), ((rx1+rx2)/2, ry1), (rx2, ry1),
            (rx1, (ry1+ry2)/2),             (rx2, (ry1+ry2)/2),
            (rx1, ry2), ((rx1+rx2)/2, ry2), (rx2, ry2)
        ]
        
        for i, (hx, hy) in enumerate(handles):
            if abs(nx - hx) < 0.04 and abs(ny - hy) < 0.04:
                if i == 0: self.video_label.config(cursor='top_left_corner')
                elif i == 1: self.video_label.config(cursor='sb_v_double_arrow')
                elif i == 2: self.video_label.config(cursor='top_right_corner')
                elif i == 3: self.video_label.config(cursor='sb_h_double_arrow')
                elif i == 4: self.video_label.config(cursor='sb_h_double_arrow')
                elif i == 5: self.video_label.config(cursor='bottom_left_corner')
                elif i == 6: self.video_label.config(cursor='sb_v_double_arrow')
                elif i == 7: self.video_label.config(cursor='bottom_right_corner')
                return
                
        if rx1 < nx < rx2 and ry1 < ny < ry2:
            self.video_label.config(cursor='fleur')
            return
            
        self.video_label.config(cursor='crosshair')

    def _update_video(self):
        global current_fps, current_live
        
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            dets = list(latest_dets)

        if frame is not None:
            h, w = frame.shape[:2]
            fw = self.video_label.winfo_width()
            fh = self.video_label.winfo_height()
            if fw > 1 and fh > 1:
                scale = min(fw/w, fh/h)
                nw, nh = int(w*scale), int(h*scale)
                resized = cv2.resize(frame, (nw, nh))
                canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
                ox = (fw - nw) // 2
                oy = (fh - nh) // 2
                canvas[oy:oy+nh, ox:ox+nw] = resized
                self._video_ox = ox
                self._video_oy = oy
                self._video_nw = nw
                self._video_nh = nh

                rx1, ry1, rx2, ry2 = self._roi
                px1, py1 = ox + int(rx1 * nw), oy + int(ry1 * nh)
                px2, py2 = ox + int(rx2 * nw), oy + int(ry2 * nh)
                
                cv2.rectangle(canvas, (px1, py1), (px2, py2), CV_ORANGE, 2)
                cv2.putText(canvas, "Focus Area", (px1+5, py1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CV_ORANGE, 2)
                
                handles = [(px1, py1), ((px1+px2)//2, py1), (px2, py1), (px1, (py1+py2)//2), (px2, (py1+py2)//2), (px1, py2), ((px1+px2)//2, py2), (px2, py2)]
                for hx, hy in handles:
                    cv2.circle(canvas, (hx, hy), 5, CV_ORANGE, -1)

                for (bx1,by1,bx2,by2,conf) in dets:
                    cx1 = ox + int(bx1 * nw / w)
                    cy1 = oy + int(by1 * nh / h)
                    cx2 = ox + int(bx2 * nw / w)
                    cy2 = oy + int(by2 * nh / h)
                    
                    cv2.rectangle(canvas,(cx1,cy1),(cx2,cy2), CV_GREEN, 2)
                    lbl = f"PERSON {conf:.0%}"
                    (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                    cv2.rectangle(canvas,(cx1,cy1-th-8),(cx1+tw+6,cy1), CV_GREEN, -1)
                    cv2.putText(canvas,lbl,(cx1+3,cy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                
                line_cy = oy + int(self._line_pct * nh)
                if py1 <= line_cy <= py2:
                    cv2.line(canvas,(px1,line_cy),(px2,line_cy), CV_YELLOW, 2)
                    cv2.putText(canvas,'OUT',(px1+8,line_cy-6),cv2.FONT_HERSHEY_SIMPLEX,0.6, CV_RED, 2)
                    cv2.putText(canvas,'IN',(px1+8,line_cy+22),cv2.FONT_HERSHEY_SIMPLEX,0.6, CV_GREEN, 2)
                
                frame = canvas
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.imgtk = imgtk
        else:
            self.video_label.config(image='')

        self.lbl_fps.config(text=f"{current_fps} FPS")
        self.lbl_net.config(text=str(current_live))

        self.after(33, self._update_video)

    def _update_stats(self):
        self.lbl_in.config(text=str(count_in))
        self.lbl_out.config(text=str(count_out))
        
        elapsed = int(time.time() - self._start_time)
        h,r = divmod(elapsed,3600)
        m,s = divmod(r,60)
        self.lbl_uptime.config(text=f"{h:02d}:{m:02d}:{s:02d}")

    def _on_count(self):
        global count_in, count_out, last_reset, tracked
        today = datetime.now().date()
        if today != last_reset:
            count_in = count_out = 0
            tracked = {}
            last_reset = today
            self.lbl_reset.config(text=f"🔄  Reset ทุกเที่ยงคืน · {self._today_str()}")
            
        save_log()
        self.after(0, self._update_stats)

    def _update_clock(self):
        self.lbl_clock.config(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    def _today_str(self):
        now = datetime.now()
        months = ['ม.ค.','ก.พ.','มี.ค.','เม.ย.','พ.ค.','มิ.ย.','ก.ค.','ส.ค.','ก.ย.','ต.ค.','พ.ย.','ธ.ค.']
        return f"{now.day} {months[now.month-1]} {now.year-1957+2500}"

    def _check_midnight_reset(self):
        global count_in, count_out, last_reset, tracked
        today = datetime.now().date()
        if today != last_reset:
            count_in = count_out = 0
            tracked = {}
            last_reset = today
            save_log()
            self._update_stats()
            self.lbl_reset.config(text=f"🔄  Reset ทุกเที่ยงคืน · {self._today_str()}")
        self.after(60000, self._check_midnight_reset)

    def _do_scan(self, auto_start=False):
        global running, cam_thread
        self.btn_scan.config(text="กำลังสแกน...", state='disabled')
        self.combo.config(state='disabled')
        def _t():
            if cam_thread and cam_thread.is_alive():
                running = False
                cam_thread.join(timeout=5)
            time.sleep(0.5)
            cams = scan_cameras()
            self.after(0, lambda: self._on_scan_done(cams, auto_start))
        threading.Thread(target=_t, daemon=True).start()

    def _on_scan_done(self, cameras, auto_start=False):
        self._cameras = cameras
        self.btn_scan.config(text="⟳  สแกน [R]", state='normal')
        if not cameras:
            self.lbl_cam_status.config(text="⚠ ไม่พบกล้อง", fg="#ef4444")
            return
        self.combo.config(values=[c['name'] for c in cameras], state='readonly')
        self.lbl_cam_status.config(text=f"พบ {len(cameras)} กล้อง", fg=TEXT_DIM)
        self.combo.current(0)
        if auto_start:
            self._start_camera(cameras[0])

    def _on_cam_selected(self, e=None):
        idx = self.combo.current()
        if 0 <= idx < len(self._cameras):
            self._start_camera(self._cameras[idx])

    def _start_camera(self, cam):
        global running, cam_thread, current_cam, latest_frame
        self.combo.config(state='disabled')
        self.btn_scan.config(state='disabled')
        self.lbl_cam_status.config(text="⏳ กำลังเริ่ม...", fg=C_YELLOW)
        with lock:
            latest_frame = None
        if cam_thread and cam_thread.is_alive():
            running = False
            cam_thread.join(timeout=8)
        current_cam = cam
        running = True
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        cam_thread = threading.Thread(
            target=camera_loop, args=(cam, self._on_count, screen_w, screen_h), daemon=True)
        cam_thread.start()
        self.after(2000, lambda: self._post_start(cam))

    def _post_start(self, cam):
        self.combo.config(state='readonly')
        self.btn_scan.config(state='normal')
        self.lbl_cam_status.config(text=f"● {cam['name']}", fg=C_IN)

    def _open_settings(self):
        SettingsDialog(self)

    def quit_app(self):
        global running
        running = False
        self.destroy()

if __name__ == '__main__':
    os.makedirs('/home/pi/people_counter', exist_ok=True)
    app = PeopleCounterApp()
    app.mainloop()
