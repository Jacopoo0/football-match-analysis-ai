import cv2, numpy as np, supervision as sv, threading, time, warnings, requests
from collections import defaultdict, deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort as BoTSORT
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker
from homography import HomographyMapper
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Percorsi ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
TEAM_JSON  = BASE_DIR / "team_colors.json"
MODEL_PT   = BASE_DIR / "models" / "soccana_best.pt"
OUTPUT_MP4 = BASE_DIR / "output_football_analysis.mp4"

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_SECONDS      = 60
PLAYER_CONF      = 0.15        # basso: recupera giocatori distanti / coperti
PLAYER_IOU       = 0.55
TEAM_HISTORY_LEN = 45          # più storia = classificazione squadra più stabile
INFER_SIZE       = 640
SAVE_VIDEO       = True
SHOW_PREVIEW     = False

FRAME_W = 960
FRAME_H = 540
PANEL_W = 380

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Palette DAZN dark ──────────────────────────────────────────────────────────
_BG         = ( 6,   8,  12)
_PANEL_BG   = ( 9,  11,  17)
_CARD_BG    = (14,  18,  28)
_CARD_ALT   = (18,  23,  35)
_BORDER     = (28,  36,  54)
_DIVIDER    = (20,  26,  40)
_WHITE      = (245, 247, 250)
_TEXT_MUTED = (105, 120, 148)
_TEXT_FAINT = ( 50,  62,  86)
_ACCENT     = (235,  55,  35)   # rosso DAZN
_T0         = ( 55, 135, 255)   # blu
_T1         = (255,  65,  65)   # rosso
_REF        = (185, 145, 255)   # viola
_BALL_C     = (255, 210,  45)   # giallo

def _bgr(c): return (c[2], c[1], c[0])
T0_BGR   = _bgr(_T0)
T1_BGR   = _bgr(_T1)
REF_BGR  = _bgr(_REF)
BALL_BGR = _bgr(_BALL_C)
BG_BGR   = _bgr(_BG)

# ── CLAHE preprocessor ────────────────────────────────────────────────────────
_clahe          = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_sharpen_kernel = np.array([[0, -0.5, 0],[-0.5, 3, -0.5],[0, -0.5, 0]], np.float32)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(frame, -1, _sharpen_kernel)

# ── Font cache ─────────────────────────────────────────────────────────────────
_FONT_CACHE: dict = {}

def _load_font(size: int, bold=False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    urls = {
        True:  "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Bold.otf",
        False: "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.otf",
    }
    cache_dir = BASE_DIR / ".font_cache"
    cache_dir.mkdir(exist_ok=True)
    fname = cache_dir / ("Inter-Bold.otf" if bold else "Inter-Regular.otf")
    if not fname.exists():
        try:
            r = requests.get(urls[bold], timeout=10)
            fname.write_bytes(r.content)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
            return _FONT_CACHE[key]
    try:
        fnt = ImageFont.truetype(str(fname), size)
    except Exception:
        fnt = ImageFont.load_default()
    _FONT_CACHE[key] = fnt
    return fnt

def _preload_fonts():
    for sz in [8,9,10,11,12,13,14,16,18,20,22,24,28,32,36,40,48]:
        _load_font(sz, False); _load_font(sz, True)

# ── Pillow helpers ─────────────────────────────────────────────────────────────
def _t(d, text, xy, size, color=_WHITE, bold=False, anchor="lt"):
    d.text(xy, str(text), font=_load_font(size, bold), fill=color, anchor=anchor)

def _card(d, x1, y1, x2, y2, fill=_CARD_BG, border=_BORDER, radius=8):
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill,
                        outline=border, width=1)

def _hbar_split(d, x, y, w, h, pct0, c0, c1, radius=3):
    d.rounded_rectangle([x,y,x+w,y+h], radius=radius, fill=_DIVIDER)
    split = max(radius*2, min(int(w*pct0/100), w-radius*2))
    d.rounded_rectangle([x,      y,x+split,y+h], radius=radius, fill=c0)
    d.rounded_rectangle([x+split,y,x+w,    y+h], radius=radius, fill=c1)

def _hbar_single(d, x, y, w, h, pct, color, radius=3):
    d.rounded_rectangle([x,y,x+w,y+h], radius=radius, fill=_DIVIDER)
    fw = max(radius*2, int(w*pct/100))
    d.rounded_rectangle([x,y,x+fw,y+h], radius=radius, fill=color)

# ── Section label ──────────────────────────────────────────────────────────────
def _section(d, label, y, M, W):
    _t(d, label, (M, y), 8, _TEXT_MUTED, bold=True)
    return y + 14

# ── Panel DAZN ─────────────────────────────────────────────────────────────────
def build_panel(stats, sec, team_counts, frame_idx, max_frames) -> np.ndarray:
    W, H = PANEL_W, FRAME_H
    img  = Image.new("RGB", (W, H), _PANEL_BG)
    d    = ImageDraw.Draw(img)
    M    = 14

    # ── Header
    d.rectangle([0, 0, W, 54], fill=_BG)
    d.rectangle([0, 0, 4, 54], fill=_ACCENT)
    mins  = int(sec) // 60
    secs_ = int(sec) % 60
    _t(d, "MATCH ANALYSIS", (M+6, 8),  10, _TEXT_MUTED, bold=True)
    _t(d, f"{mins:02d}:{secs_:02d}",    (W-M,  8),  32, _WHITE, bold=True, anchor="rt")
    pct_done = int(frame_idx / max(max_frames,1) * 100)
    _t(d, f"{pct_done}%", (M+6, 38), 9, _TEXT_FAINT)
    prog_w = int(W * frame_idx / max(max_frames,1))
    d.rectangle([0, 50, W,      54], fill=_DIVIDER)
    d.rectangle([0, 50, prog_w, 54], fill=_ACCENT)
    y = 62

    # ── Possesso
    y = _section(d, "BALL POSSESSION", y, M, W)
    p0, p1   = stats.possession_pct()
    rp0, rp1 = stats.recent_possession()
    cx = W // 2
    _t(d, f"{p0:.0f}%", (cx-10, y), 38, _T0, bold=True, anchor="rt")
    _t(d, "–",           (cx,    y+12), 12, _TEXT_FAINT, anchor="mt")
    _t(d, f"{p1:.0f}%", (cx+10, y), 38, _T1, bold=True)
    y += 44
    _t(d, "TEAM 0", (M,   y), 8, _T0, bold=True)
    _t(d, "TEAM 1", (W-M, y), 8, _T1, bold=True, anchor="rt")
    y += 10
    _hbar_split(d, M, y, W-M*2, 6, p0, _T0, _T1, radius=3)
    y += 12
    _t(d, f"Last 5s  {rp0:.0f}% – {rp1:.0f}%", (cx, y), 8, _TEXT_MUTED, anchor="mt")
    y += 14
    d.rectangle([M, y, W-M, y+1], fill=_DIVIDER); y += 8

    # ── Giocatori
    y = _section(d, "PLAYERS ON FIELD", y, M, W)
    hw  = (W - M*2 - 8) // 2
    x2s = M + hw + 8
    for xi, tid, col in [(M, 0, _T0), (x2s, 1, _T1)]:
        cnt = team_counts.get(tid, 0)
        _card(d, xi, y, xi+hw, y+48, fill=_CARD_BG, radius=6)
        d.rectangle([xi, y, xi+hw, y+3], fill=col)
        _t(d, str(cnt),         (xi+hw//2, y+6),  26, col,        bold=True, anchor="mt")
        _t(d, f"TEAM {tid}",    (xi+hw//2, y+36),  8, _TEXT_MUTED, anchor="mt")
    y += 56
    d.rectangle([M, y, W-M, y+1], fill=_DIVIDER); y += 8

    # ── Passaggi
    y = _section(d, "PASSES", y, M, W)
    ps0  = stats.passes.get(0, 0)
    ps1  = stats.passes.get(1, 0)
    totP = max(ps0+ps1, 1)
    _card(d, M, y, W-M, y+36, fill=_CARD_BG, radius=6)
    _t(d, str(ps0), (M+16,   y+8), 20, _T0, bold=True)
    _t(d, "T0",     (M+16,   y+29), 8, _TEXT_MUTED)
    _t(d, str(ps1), (W-M-16, y+8), 20, _T1, bold=True, anchor="rt")
    _t(d, "T1",     (W-M-16, y+29), 8, _TEXT_MUTED, anchor="rt")
    bw = W-M*2-80; bx = M+40
    _hbar_split(d, bx, y+18, bw, 4, ps0/totP*100, _T0, _T1, radius=2)
    y += 44
    d.rectangle([M, y, W-M, y+1], fill=_DIVIDER); y += 8

    # ── Velocità
    y = _section(d, "SPEED  avg / max", y, M, W)
    s0, s1   = stats.avg_speed_kmh()
    mx0, mx1 = stats.max_speed_kmh()
    for xi, spd, mspd, col in [(M, s0, mx0, _T0), (x2s, s1, mx1, _T1)]:
        _card(d, xi, y, xi+hw, y+50, fill=_CARD_BG, radius=6)
        d.rectangle([xi, y, xi+hw, y+3], fill=col)
        _t(d, f"{spd:.1f}",   (xi+10, y+5),  20, col,        bold=True)
        _t(d, "km/h",          (xi+10, y+27),  8, _TEXT_MUTED)
        _t(d, f"max {mspd:.1f}", (xi+10, y+38), 8, _TEXT_FAINT)
    y += 58
    d.rectangle([M, y, W-M, y+1], fill=_DIVIDER); y += 8

    # ── Distanza
    y = _section(d, "DISTANCE COVERED", y, M, W)
    d0, d1  = stats.distance_meters()
    max_d   = max(d0, d1, 1.0)
    bw_full = W-M*2-60
    for dist, col, label in [(d0, _T0, "T0"), (d1, _T1, "T1")]:
        _t(d, label, (M, y+2), 8, col, bold=True)
        _hbar_single(d, M+22, y, bw_full, 7, dist/max_d*100, col, radius=3)
        _t(d, f"{dist/1000:.2f} km", (W-M, y+2), 8, col, bold=True, anchor="rt")
        y += 16
    y += 4
    d.rectangle([M, y, W-M, y+1], fill=_DIVIDER); y += 6

    # ── Footer
    refn = team_counts.get(2, 0)
    unkn = team_counts.get(-1, 0)
    _t(d, f"REF {refn}   UNK {unkn}", (M, y+2), 8, _TEXT_FAINT)
    _t(d, f"{frame_idx} / {max_frames}", (W-M, y+2), 8, _TEXT_FAINT, anchor="rt")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Etichette giocatori — stile broadcast ────────────────────────────────────
def draw_player(frame, x1, y1, x2, y2, tid, team_id, color_bgr):
    cx  = (x1 + x2) // 2
    w   = x2 - x1
    h   = y2 - y1

    # --- bounding box sottile colorata
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
    # angolini più spessi (broadcast style)
    seg = max(4, min(w//4, 10))
    cv2.line(frame, (x1,      y1), (x1+seg,  y1), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x1,      y1), (x1,      y1+seg), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x2,      y1), (x2-seg,  y1), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x2,      y1), (x2,      y1+seg), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x1,      y2), (x1+seg,  y2), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x1,      y2), (x1,      y2-seg), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x2,      y2), (x2-seg,  y2), color_bgr, 2, cv2.LINE_AA)
    cv2.line(frame, (x2,      y2), (x2,      y2-seg), color_bgr, 2, cv2.LINE_AA)

    # --- dot sul piedistallo (ellisse ombra)
    bw = max(6, int(w * 0.38))
    bh = max(2, int(w * 0.09))
    ov = frame.copy()
    cv2.ellipse(ov, (cx, y2), (bw, bh), 0, 0, 360, color_bgr, -1)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)

    # --- pill label compatta sopra la testa
    if team_id == 2:
        label = "REF"
    else:
        label = f"#{tid}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.28
    thick      = 1
    (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 3
    lx  = cx - lw // 2
    ly  = y1 - 5
    # sfondo pill
    cv2.rectangle(frame,
                  (lx-pad,    ly-lh-pad),
                  (lx+lw+pad, ly+pad),
                  (5, 7, 12), -1)
    # bordo colorato pill
    cv2.rectangle(frame,
                  (lx-pad,    ly-lh-pad),
                  (lx+lw+pad, ly+pad),
                  color_bgr, 1)
    cv2.putText(frame, label, (lx, ly),
                font, scale, (238, 242, 248), thick, cv2.LINE_AA)


def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    r = max(5, (x2-x1)//2)
    cv2.circle(frame, (cx, cy), r,    BALL_BGR,       -1)
    cv2.circle(frame, (cx, cy), r,    (160, 130, 15),  1)
    cv2.circle(frame, (cx, cy), r//3, (255, 248, 200), -1)


# ── Canvas compositing — minimap centrata verticalmente ───────────────────────
def build_canvas(frame_img, panel_img, minimap_img=None):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    total_h = max(fh, ph)
    total_w = fw + pw + 2
    C = np.full((total_h, total_w, 3), BG_BGR, dtype=np.uint8)
    fy = (total_h - fh) // 2
    C[fy:fy+fh, 0:fw]             = frame_img
    C[(total_h-ph)//2:(total_h-ph)//2+ph, fw+2:fw+2+pw] = panel_img
    C[:, fw:fw+2] = _bgr(_BORDER)

    if minimap_img is not None:
        mh, mw = minimap_img.shape[:2]
        # Centrata orizzontalmente sul frame, 12px dal bordo inferiore
        mx = (fw - mw) // 2
        my = fy + fh - mh - 12
        if my >= fy and mx >= 0:
            # Bordo sottile + blend
            bordered = cv2.copyMakeBorder(minimap_img, 1,1,1,1,
                                          cv2.BORDER_CONSTANT, value=_bgr(_BORDER))
            bh2, bw2 = bordered.shape[:2]
            roi     = C[my-1:my-1+bh2, mx-1:mx-1+bw2].astype(np.float32)
            mm_f    = bordered.astype(np.float32)
            blended = (mm_f*0.90 + roi*0.10).astype(np.uint8)
            C[my-1:my-1+bh2, mx-1:mx-1+bw2] = blended
    return C


# ── Worker ────────────────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    def __init__(self, model, team_classifier, tracker, stats, homography, fps):
        super().__init__(daemon=True)
        self.model      = model
        self.tc         = team_classifier
        self.tracker    = tracker
        self.stats      = stats
        self.homography = homography
        self.fps        = fps
        self.team_history      = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
        self._in_frame         = None
        self._in_lock          = threading.Lock()
        self._in_event         = threading.Event()
        self.result_frame      = None
        self.result_lock       = threading.Lock()
        self._stop             = False
        self._sec = self._fidx = self._maxf = 0
        self._last_ball_center = None
        self._ball_lost_frames = 0

    def submit(self, frame, sec, fidx, maxf):
        with self._in_lock:
            self._in_frame = frame.copy()
            self._sec, self._fidx, self._maxf = sec, fidx, maxf
        self._in_event.set()

    def stop(self):
        self._stop = True; self._in_event.set()

    def run(self):
        while not self._stop:
            self._in_event.wait(); self._in_event.clear()
            if self._stop: break
            with self._in_lock:
                frame = self._in_frame.copy()
                sec, fidx, maxf = self._sec, self._fidx, self._maxf

            team_counts = {0:0, 1:0, -1:0, 2:0}
            players_for_stats = []
            ball_center       = None
            field_positions   = {}

            try:
                frame_proc = preprocess_frame(frame)
                self.homography.update_frame(frame_proc)

                # ── Detection giocatori
                results_p = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=PLAYER_CONF,
                    iou=PLAYER_IOU, device="cuda", verbose=False,
                    classes=[CLS_PLAYER, CLS_REFEREE],
                    augment=True
                )[0]
                results_b = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=0.04,
                    iou=0.30, device="cuda", verbose=False,
                    classes=[CLS_BALL]
                )[0]

                player_det = sv.Detections.from_ultralytics(results_p)
                ball_det   = sv.Detections.from_ultralytics(results_b)

                # ── Secondo pass sulla metà superiore (giocatori lontani/piccoli)
                fh_half = frame_proc.shape[0] // 2
                top_half = frame_proc[:fh_half, :]
                results_far = self.model.predict(
                    top_half, imgsz=INFER_SIZE, conf=0.10,
                    iou=0.50, device="cuda", verbose=False,
                    classes=[CLS_PLAYER]
                )[0]
                if len(results_far.boxes) > 0:
                    det_far = sv.Detections.from_ultralytics(results_far)
                    # Correggi coordinate Y: erano relative alla metà superiore
                    det_far.xyxy[:, 1] += 0   # y1 già corretto (crop dal top)
                    det_far.xyxy[:, 3] += 0   # y2 idem
                    # merge NMS manuale: tieni solo box far non già coperti
                    if len(player_det) > 0:
                        player_det = sv.Detections.merge([player_det, det_far])
                    else:
                        player_det = det_far

                if len(player_det) > 0:
                    areas = ((player_det.xyxy[:,2]-player_det.xyxy[:,0]) *
                             (player_det.xyxy[:,3]-player_det.xyxy[:,1]))
                    player_det = player_det[areas > 300]

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
                tracks = np.empty((0, 8)); ball_det = sv.Detections.empty()

            # ── Palla
            if ball_det is not None and len(ball_det) > 0:
                best = int(np.argmax(ball_det.confidence))
                bx1,by1,bx2,by2 = map(int, ball_det.xyxy[best])
                ball_center = ((bx1+bx2)//2, (by1+by2)//2)
                self._last_ball_center = ball_center
                self._ball_lost_frames = 0
                draw_ball(frame, bx1, by1, bx2, by2)
                self.homography.update_ball(ball_center)
            else:
                self._ball_lost_frames += 1
                if self._ball_lost_frames <= 20 and self._last_ball_center:
                    ball_center = self._last_ball_center

            # ── Tracks
            for track in tracks:
                if len(track) < 7: continue
                x1,y1,x2,y2 = map(int, track[:4])
                tid          = int(track[4])
                class_id     = int(track[6])
                cx, cy       = (x1+x2)//2, (y1+y2)//2

                if class_id == CLS_REFEREE:
                    team_id = 2; color = REF_BGR
                else:
                    raw, _ = self.tc.classify_player(frame, (x1, y1, x2, y2))
                    if raw != -1:
                        self.team_history[tid].append(raw)
                    team_id = (Counter(self.team_history[tid]).most_common(1)[0][0]
                               if len(self.team_history[tid]) >= 3 else raw)
                    if   team_id == 0: color = T0_BGR
                    elif team_id == 1: color = T1_BGR
                    else:              team_id = -1; color = _bgr((115, 128, 152))

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame, x1, y1, x2, y2, tid, team_id, color)
                players_for_stats.append((cx, cy, team_id, tid))
                fp = self.homography.pixel_to_field((cx, cy))
                if fp: field_positions[tid] = (team_id, fp)

            self.stats._last_players = players_for_stats
            self.stats.update(ball_center, players_for_stats)
            self.homography.update_players(field_positions)
            # Passa scala pixel/metro allo StatsTracker se calibrato
            if self.homography.px_per_m is not None:
                self.stats.set_scale(self.homography.px_per_m)
            minimap = self.homography.render_minimap()

            frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
            panel       = build_panel(self.stats, sec, team_counts, fidx, maxf)
            canvas      = build_canvas(frame_small, panel, minimap)

            with self.result_lock:
                self.result_frame = canvas

    def get_canvas(self):
        with self.result_lock:
            return self.result_frame


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    for path, label in [
        (VIDEO_PATH, "Video input"),
        (TEAM_JSON,  "team_colors.json"),
        (MODEL_PT,   "soccana_best.pt"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}\n  {path}"); return

    print("Caricamento font..."); _preload_fonts()
    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    model.predict(np.zeros((640,640,3), dtype=np.uint8),
                  imgsz=INFER_SIZE, device="cuda", verbose=False)

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    reid_weights = BASE_DIR / "models" / "osnet_ain_x1_0_msmt17.pt"

    tracker = BoTSORT(
        reid_weights      = reid_weights,
        device            = "0",
        half              = False,
        track_high_thresh = 0.25,
        track_low_thresh  = 0.10,
        new_track_thresh  = 0.10,
        track_buffer      = 180,
        match_thresh      = 0.90,
        cmc_method        = "sof",
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}"); return

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
        cv2.resizeWindow("Football Analysis", FRAME_W + PANEL_W + 4, FRAME_H)

    print(f"Avvio  {max_frames} frame @ {fps:.1f} fps")
    frame_idx = 0; last_canvas = None; spf = 1.0/fps; t_last = time.time()

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        worker.submit(frame, sec, frame_idx, max_frames)
        canvas = worker.get_canvas()
        if canvas is not None: last_canvas = canvas
        if last_canvas is not None:
            if SAVE_VIDEO:
                if out is None:
                    h, w   = last_canvas.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w,h))
                out.write(last_canvas)
            if SHOW_PREVIEW:
                cv2.imshow("Football Analysis", last_canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
        elapsed = time.time() - t_last
        if spf - elapsed > 0: time.sleep(spf - elapsed)
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

    worker.stop(); cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO: print(f"\nSalvato: {OUTPUT_MP4}")

if __name__ == "__main__":
    main()