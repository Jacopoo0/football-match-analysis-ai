import cv2
import numpy as np
import supervision as sv
import threading
import time
import warnings
from collections import defaultdict, deque, Counter
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.strongsort.strongsort import StrongSort
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker
from homography import HomographyMapper
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
TEAM_JSON  = BASE_DIR / "team_colors.json"
MODEL_PT   = BASE_DIR / "models" / "soccana_best.pt"
OUTPUT_MP4 = BASE_DIR / "output_football_analysis.mp4"

MAX_SECONDS      = 60
PLAYER_CONF      = 0.25
PLAYER_IOU       = 0.55
TEAM_HISTORY_LEN = 45
INFER_SIZE       = 640
SAVE_VIDEO       = True
SHOW_PREVIEW     = False
FRAME_W          = 960
FRAME_H          = 540

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

T0_BGR   = (255, 135,  55)
T1_BGR   = ( 65,  65, 255)
REF_BGR  = (255, 145, 185)
BALL_BGR = ( 45, 210, 255)
UNK_BGR  = (152, 128, 115)
HUD_BG   = (10,  12,  18)

_clahe          = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_sharpen_kernel = np.array([[0,-0.4,0],[-0.4,2.6,-0.4],[0,-0.4,0]], np.float32)

def preprocess_frame(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = _clahe.apply(l)
    frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(frame, -1, _sharpen_kernel)

def _alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color)
    img[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)

def draw_hud(img, stats, sec, team_counts, frame_idx, max_frames):
    H, W  = img.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    fs    = 0.42
    fmed  = 0.52
    BAR_H = 34

    _alpha_rect(img, 0, 0, W, BAR_H, HUD_BG, alpha=0.72)
    cv2.line(img, (0, BAR_H), (W, BAR_H), (35, 55, 235), 1)

    mins  = int(sec) // 60
    secs_ = int(sec) % 60
    cv2.putText(img, f"{mins:02d}:{secs_:02d}", (12, 23), font, fmed,
                (245, 247, 250), 1, cv2.LINE_AA)

    p0, p1  = stats.possession_pct()
    BAR_W   = 220
    BAR_TH  = 6
    bx      = W // 2 - BAR_W // 2
    by      = BAR_H // 2 - BAR_TH // 2
    t0_lbl  = f"T0  {p0:.0f}%"
    t1_lbl  = f"{p1:.0f}%  T1"
    (lw0,_),_ = cv2.getTextSize(t0_lbl, font, fs, 1)
    cv2.putText(img, t0_lbl, (bx - lw0 - 8, 22), font, fs, T0_BGR, 1, cv2.LINE_AA)
    cv2.putText(img, t1_lbl, (bx + BAR_W + 8, 22), font, fs, T1_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + BAR_W, by + BAR_TH), (40, 45, 55), -1)
    split = max(2, min(int(BAR_W * p0 / 100), BAR_W - 2))
    cv2.rectangle(img, (bx,       by), (bx + split,  by + BAR_TH), T0_BGR, -1)
    cv2.rectangle(img, (bx+split, by), (bx + BAR_W,  by + BAR_TH), T1_BGR, -1)

    prog = int(W * frame_idx / max(max_frames, 1))
    cv2.rectangle(img, (0, BAR_H - 2), (W,    BAR_H), (30, 35, 45), -1)
    cv2.rectangle(img, (0, BAR_H - 2), (prog, BAR_H), (35, 55, 235), -1)

    ps0 = stats.passes.get(0, 0)
    ps1 = stats.passes.get(1, 0)
    pass_str = f"PASS  {ps0} | {ps1}"
    (pw,_),_ = cv2.getTextSize(pass_str, font, fs, 1)
    cv2.putText(img, pass_str, (W - pw - 14, 22), font, fs,
                (180, 190, 210), 1, cv2.LINE_AA)

    MM_W, MM_H = 240, 156
    MM_PAD     = 10
    d0, d1     = stats.distance_meters()
    s0, s1     = stats.avg_speed_kmh()
    rp0, rp1   = stats.recent_possession()
    lines = [
        (f"DIST  {d0/1000:.2f} | {d1/1000:.2f} km", T0_BGR if d0 >= d1 else T1_BGR),
        (f"SPD   {s0:.1f} | {s1:.1f} km/h",          T0_BGR if s0 >= s1 else T1_BGR),
        (f"5s    {rp0:.0f}% | {rp1:.0f}%",           (160, 170, 190)),
    ]
    block_h = len(lines) * 17 + 6
    info_y  = H - MM_H - MM_PAD - 4
    _alpha_rect(img, MM_PAD, info_y - block_h, MM_W + MM_PAD, info_y, HUD_BG, 0.60)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt,
                    (MM_PAD + 6, info_y - block_h + 14 + i * 17),
                    font, fs, col, 1, cv2.LINE_AA)

def draw_player(frame, x1, y1, x2, y2, team_id, color_bgr):
    w   = x2 - x1
    cx  = (x1 + x2) // 2
    seg = max(4, min(w // 3, 12))
    for (px_, py_), (dx, dy) in [
        ((x1,y1),(+seg,0)),((x1,y1),(0,+seg)),
        ((x2,y1),(-seg,0)),((x2,y1),(0,+seg)),
        ((x1,y2),(+seg,0)),((x1,y2),(0,-seg)),
        ((x2,y2),(-seg,0)),((x2,y2),(0,-seg)),
    ]:
        cv2.line(frame, (px_,py_), (px_+dx,py_+dy), color_bgr, 2, cv2.LINE_AA)
    dot_r = max(3, min(w // 7, 5))
    foot  = (cx, y2 + dot_r)
    cv2.circle(frame, foot, dot_r + 2, (0,0,0),      -1, cv2.LINE_AA)
    cv2.circle(frame, foot, dot_r,     color_bgr,    -1, cv2.LINE_AA)
    cv2.circle(frame, foot, dot_r,     (255,255,255),  1, cv2.LINE_AA)
    if team_id == 2:
        cv2.putText(frame, "R", (cx - 4, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, REF_BGR, 1, cv2.LINE_AA)

def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    r      = max(6, (x2-x1)//2 + 2)
    glow   = frame.copy()
    cv2.circle(glow, (cx,cy), r+5, BALL_BGR, -1)
    cv2.addWeighted(glow, 0.20, frame, 0.80, 0, frame)
    cv2.circle(frame, (cx,cy), r,           BALL_BGR,       -1, cv2.LINE_AA)
    cv2.circle(frame, (cx,cy), r,           (80,140,160),    1, cv2.LINE_AA)
    cv2.circle(frame, (cx-r//3,cy-r//3), max(2,r//3), (255,252,210), -1, cv2.LINE_AA)

def overlay_minimap(frame, minimap_img):
    MM_W, MM_H = 240, 156
    PAD        = 10
    H          = frame.shape[0]
    mm         = cv2.resize(minimap_img, (MM_W, MM_H), interpolation=cv2.INTER_AREA)
    mm_b       = cv2.copyMakeBorder(mm, 1,1,1,1, cv2.BORDER_CONSTANT, value=(40,48,68))
    mh, mw     = mm_b.shape[:2]
    x1 = PAD - 1
    y1 = H - mh - PAD + 1
    if y1 < 0 or x1 < 0:
        return
    roi     = frame[y1:y1+mh, x1:x1+mw].astype(np.float32)
    blended = (mm_b.astype(np.float32)*0.93 + roi*0.07).astype(np.uint8)
    frame[y1:y1+mh, x1:x1+mw] = blended


class InferenceWorker(threading.Thread):
    def __init__(self, model, team_classifier, tracker, stats, homography, fps):
        super().__init__(daemon=True)
        self.model        = model
        self.tc           = team_classifier
        self.tracker      = tracker
        self.stats        = stats
        self.homography   = homography
        self.fps          = fps
        self.team_history = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
        self._in_frame    = None
        self._in_lock     = threading.Lock()
        self._in_event    = threading.Event()
        self.result_frame = None
        self.result_lock  = threading.Lock()
        self._stop        = False
        self._sec = self._fidx = self._maxf = 0
        self._last_ball_center = None
        self._ball_lost_frames = 0

    def submit(self, frame, sec, fidx, maxf):
        with self._in_lock:
            self._in_frame = frame.copy()
            self._sec, self._fidx, self._maxf = sec, fidx, maxf
        self._in_event.set()

    def stop(self):
        self._stop = True
        self._in_event.set()

    def run(self):
        while not self._stop:
            self._in_event.wait()
            self._in_event.clear()
            if self._stop:
                break
            with self._in_lock:
                frame = self._in_frame.copy()
                sec, fidx, maxf = self._sec, self._fidx, self._maxf

            team_counts     = {0:0, 1:0, -1:0, 2:0}
            players_stats   = []
            ball_center     = None
            field_positions = {}

            try:
                frame_proc = preprocess_frame(frame)
                self.homography.update_frame(frame_proc)

                res_p = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=PLAYER_CONF,
                    iou=PLAYER_IOU, device="cuda", verbose=False,
                    classes=[CLS_PLAYER, CLS_REFEREE]
                )[0]
                res_b = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=0.05,
                    iou=0.30, device="cuda", verbose=False,
                    classes=[CLS_BALL]
                )[0]

                player_det = sv.Detections.from_ultralytics(res_p)
                ball_det   = sv.Detections.from_ultralytics(res_b)

                fh_half = frame_proc.shape[0] // 2
                res_far = self.model.predict(
                    frame_proc[:fh_half], imgsz=INFER_SIZE, conf=0.12,
                    iou=0.50, device="cuda", verbose=False,
                    classes=[CLS_PLAYER]
                )[0]
                if len(res_far.boxes) > 0:
                    det_far    = sv.Detections.from_ultralytics(res_far)
                    player_det = sv.Detections.merge([player_det, det_far]) \
                                 if len(player_det) > 0 else det_far

                if len(player_det) > 0:
                    areas      = ((player_det.xyxy[:,2]-player_det.xyxy[:,0]) *
                                  (player_det.xyxy[:,3]-player_det.xyxy[:,1]))
                    player_det = player_det[areas > 200]

                if len(player_det) > 0:
                    dets_np = np.column_stack([
                        player_det.xyxy,
                        player_det.confidence,
                        player_det.class_id.astype(float)
                    ])
                    tracks = self.tracker.update(dets_np, frame_proc)
                else:
                    tracks = np.empty((0, 8))

            except Exception as e:
                print(f"Errore inference: {e}")
                tracks   = np.empty((0, 8))
                ball_det = sv.Detections.empty()

            frame_draw = cv2.resize(frame, (FRAME_W, FRAME_H))
            sx = FRAME_W / frame.shape[1]
            sy = FRAME_H / frame.shape[0]

            if ball_det is not None and len(ball_det) > 0:
                best       = int(np.argmax(ball_det.confidence))
                bx1 = int(ball_det.xyxy[best][0]*sx)
                by1 = int(ball_det.xyxy[best][1]*sy)
                bx2 = int(ball_det.xyxy[best][2]*sx)
                by2 = int(ball_det.xyxy[best][3]*sy)
                ball_center = ((bx1+bx2)//2, (by1+by2)//2)
                self._last_ball_center = ball_center
                self._ball_lost_frames = 0
                draw_ball(frame_draw, bx1, by1, bx2, by2)
                self.homography.update_ball(ball_center)
            else:
                self._ball_lost_frames += 1
                if self._ball_lost_frames <= 20 and self._last_ball_center:
                    ball_center = self._last_ball_center

            for track in tracks:
                if len(track) < 7:
                    continue
                x1o,y1o,x2o,y2o = map(int, track[:4])
                tid      = int(track[4])
                class_id = int(track[6])
                x1,y1,x2,y2 = int(x1o*sx),int(y1o*sy),int(x2o*sx),int(y2o*sy)
                cx_orig  = (x1o+x2o)//2
                cy_orig  = (y1o+y2o)//2

                if class_id == CLS_REFEREE:
                    team_id = 2; color = REF_BGR
                else:
                    raw, _ = self.tc.classify_player(frame, (x1o,y1o,x2o,y2o))
                    if raw != -1:
                        self.team_history[tid].append(raw)
                    h_buf   = self.team_history[tid]
                    team_id = Counter(h_buf).most_common(1)[0][0] \
                              if len(h_buf) >= 3 else raw
                    if   team_id == 0: color = T0_BGR
                    elif team_id == 1: color = T1_BGR
                    else:              team_id = -1; color = UNK_BGR

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame_draw, x1, y1, x2, y2, team_id, color)
                players_stats.append((cx_orig, cy_orig, team_id, tid))
                fp = self.homography.pixel_to_field((cx_orig, cy_orig))
                field_positions[tid] = (
                    team_id,
                    fp if fp is not None else (float(cx_orig), float(cy_orig))
                )

            self.stats.update(ball_center, players_stats)
            if self.homography.px_per_m is not None:
                self.stats.set_scale(self.homography.px_per_m)

            self.homography.update_players(field_positions)
            minimap = self.homography.render_minimap()

            draw_hud(frame_draw, self.stats, sec, team_counts, fidx, maxf)
            overlay_minimap(frame_draw, minimap)

            with self.result_lock:
                self.result_frame = frame_draw

    def get_canvas(self):
        with self.result_lock:
            return self.result_frame


def main():
    for path, label in [
        (VIDEO_PATH, "Video input"),
        (TEAM_JSON,  "team_colors.json"),
        (MODEL_PT,   "soccana_best.pt"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}  ->  {path}")
            return

    print("Caricamento modello YOLO...")
    model = YOLO(str(MODEL_PT))
    model.predict(np.zeros((640,640,3), dtype=np.uint8),
                  imgsz=INFER_SIZE, device="cuda", verbose=False)

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))

    reid_weights = BASE_DIR / "models" / "osnet_ain_x1_0_msmt17.pt"
    if not reid_weights.exists():
        reid_weights = BASE_DIR / "models" / "osnet_x0_25_msmt17.pt"

    tracker = StrongSort(
        reid_weights = reid_weights,
        device       = "0",
        half         = False,
        det_thresh   = 0.1,
        max_dist     = 0.3,
        max_iou_dist = 0.7,
        max_age      = 60,
        n_init       = 2,
        nn_budget    = 100,
        mc_lambda    = 0.995,
        ema_alpha    = 0.9,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames

    stats      = StatsTracker(fps=fps)
    homography = HomographyMapper()
    worker     = InferenceWorker(model, team_classifier, tracker,
                                  stats, homography, fps)
    worker.start()
    out = None

    if SHOW_PREVIEW:
        cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Analysis", FRAME_W, FRAME_H)

    print(f"Avvio  {max_frames} frame @ {fps:.1f} fps")
    frame_idx   = 0
    last_canvas = None
    t_last      = time.time()
    spf         = 1.0 / fps

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        worker.submit(frame, sec, frame_idx, max_frames)
        canvas = worker.get_canvas()
        if canvas is not None:
            last_canvas = canvas
            if SAVE_VIDEO:
                if out is None:
                    h_c, w_c = canvas.shape[:2]
                    for codec, ext in [("mp4v",".mp4"),("XVID",".avi"),("MJPG",".avi")]:
                        out_path = OUTPUT_MP4 if ext==".mp4" \
                                   else OUTPUT_MP4.with_suffix(ext)
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out    = cv2.VideoWriter(str(out_path), fourcc, fps, (w_c,h_c))
                        if out.isOpened():
                            print(f"VideoWriter: {codec} -> {out_path.name}")
                            break
                if out and out.isOpened():
                    out.write(canvas)
            if SHOW_PREVIEW and last_canvas is not None:
                cv2.imshow("Football Analysis", last_canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        elapsed = time.time() - t_last
        if spf - elapsed > 0:
            time.sleep(spf - elapsed)
        t_last = time.time()

        if frame_idx % 60 == 0:
            p0,p1 = stats.possession_pct()
            d0,d1 = stats.distance_meters()
            s0,s1 = stats.avg_speed_kmh()
            print(f"[{frame_idx}/{max_frames}] {sec:.1f}s | "
                  f"Poss {p0:.0f}/{p1:.0f}% | "
                  f"Pass {stats.passes[0]}/{stats.passes[1]} | "
                  f"Dist {d0:.0f}/{d1:.0f}m | "
                  f"Vel {s0:.1f}/{s1:.1f}")

    worker.stop()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO:
        print(f"\nSalvato: {OUTPUT_MP4}")

if __name__ == "__main__":
    main()