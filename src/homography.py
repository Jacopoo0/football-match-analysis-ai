"""
HomographyMapper — minimappa con camera pose estimation PnP
Integra l'approccio di Antoine Keller (github.com/antoinekeller/soccer_tracker):
  - Rileva keypoints semantici del campo (cerchio centrale, linee laterali, angoli)
  - Risolve PnP per stimare posa camera in 3D
  - Fallback a omografia 2D da Hough se PnP non converge
  - Fallback finale a GreenBBox
"""

import cv2
import numpy as np
from collections import deque

# ── Dimensioni campo FIFA standard (metri) ─────────────────────────────────────
FIELD_W  = 105.0   # lunghezza
FIELD_H  =  68.0   # larghezza

# Keypoints 3D nel sistema mondo: origine = centro campo, x→destra, z→avanti
# (coordinate coerenti con Keller: y=0 = piano campo)
KP_WORLD = {
    "right_circle":       [ 9.15, 0,   0.0],
    "left_circle":        [-9.15, 0,   0.0],
    "behind_circle":      [ 0.0,  0,   9.15],
    "front_circle":       [ 0.0,  0,  -9.15],
    "front_middle_line":  [ 0.0,  0, -34.0],
    "back_middle_line":   [ 0.0,  0,  34.0],
    "corner_back_left":   [-52.5, 0,  34.0],
    "corner_front_left":  [-52.5, 0, -34.0],
    "corner_back_right":  [ 52.5, 0,  34.0],
    "corner_front_right": [ 52.5, 0, -34.0],
}

# ── Minimap ────────────────────────────────────────────────────────────────────
MAP_W, MAP_H = 320, 208
MAP_MARGIN   = 14

_FIELD_GREEN   = ( 42, 100,  42)
_LINE_WHITE    = (220, 220, 220)
_DOT_BALL      = ( 45, 210, 255)


def _build_base_minimap():
    """Disegna il campo 2D (topview) una volta sola."""
    img = np.full((MAP_H, MAP_W, 3), _FIELD_GREEN, dtype=np.uint8)
    mx, my = MAP_MARGIN, MAP_MARGIN
    fw, fh = MAP_W - 2*mx, MAP_H - 2*my
    t = 1

    def px(x_m, z_m):
        u = int(mx + (x_m + FIELD_W/2) / FIELD_W * fw)
        v = int(my + (z_m + FIELD_H/2) / FIELD_H * fh)
        return (u, v)

    # Strisce erba
    for i in range(7):
        x0 = mx + i * fw // 7
        x1 = mx + (i+1) * fw // 7
        col = (38, 92, 38) if i % 2 == 0 else (46, 108, 46)
        img[my:my+fh, x0:x1] = col

    # Bordo campo
    cv2.rectangle(img, px(-FIELD_W/2, -FIELD_H/2), px(FIELD_W/2, FIELD_H/2), _LINE_WHITE, t)
    # Linea centrocampo
    cv2.line(img, px(0, -FIELD_H/2), px(0, FIELD_H/2), _LINE_WHITE, t)
    # Cerchio centrocampo
    r_px = int(9.15 / FIELD_W * fw)
    cv2.circle(img, px(0, 0), r_px, _LINE_WHITE, t)
    cv2.circle(img, px(0, 0), 2, _LINE_WHITE, -1)
    # Aree rigore sx
    cv2.rectangle(img, px(-FIELD_W/2, -20.16), px(-FIELD_W/2+16.5, 20.16), _LINE_WHITE, t)
    # Aree rigore dx
    cv2.rectangle(img, px(FIELD_W/2-16.5, -20.16), px(FIELD_W/2, 20.16), _LINE_WHITE, t)
    # Dischetti rigore
    cv2.circle(img, px(-FIELD_W/2+11, 0), 2, _LINE_WHITE, -1)
    cv2.circle(img, px( FIELD_W/2-11, 0), 2, _LINE_WHITE, -1)

    return img

_BASE_MINIMAP = _build_base_minimap()


def _field_to_map(x_m, z_m):
    """Campo (metri) → pixel minimappa."""
    u = int(MAP_MARGIN + (x_m + FIELD_W/2) / FIELD_W * (MAP_W - 2*MAP_MARGIN))
    v = int(MAP_MARGIN + (z_m + FIELD_H/2) / FIELD_H * (MAP_H - 2*MAP_MARGIN))
    return u, v


# ── Pitch line detection (da Keller) ──────────────────────────────────────────

def _intersect_polar(line1, line2):
    """Intersezione tra due linee in coord polari (rho, theta)."""
    if line1 is None or line2 is None:
        return None
    r1, t1 = line1
    r2, t2 = line2
    if abs(t1 - t2) < 1e-6:
        return None
    u = (r1 * np.sin(t2) - r2 * np.sin(t1)) / np.sin(t2 - t1)
    v = (r1 - u * np.cos(t1)) / np.sin(t1) if abs(np.sin(t1)) > 1e-6 \
        else (r2 - u * np.cos(t2)) / np.sin(t2)
    return [int(u), int(v)]


def _find_back_front_lines(gray):
    """Trova linee laterali del campo (quasi orizzontali)."""
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180/4, 300, None,
                           min_theta=75/180*np.pi, max_theta=105/180*np.pi)
    if lines is None:
        return None, None
    h = gray.shape[0]
    w = gray.shape[1]
    back_line, front_line = None, None
    back_y, front_y = 0, h
    for line in lines:
        rho, theta = line[0]
        y_mid = (rho - w/2 * np.cos(theta)) / np.sin(theta)
        if back_y < y_mid < h/2:
            back_y = y_mid; back_line = line[0]
        if h/2 < y_mid < front_y:
            front_y = y_mid; front_line = line[0]
    return back_line, front_line


def _find_main_line(gray, back_line, front_line):
    """Trova la linea verticale centrale (mezzeria)."""
    # Maschera: solo la zona dentro le linee laterali
    h, w = gray.shape
    mask = np.zeros_like(gray)
    if back_line is not None and front_line is not None:
        for j in range(w):
            yb = int((back_line[0]  - j*np.cos(back_line[1]))  / np.sin(back_line[1]))
            yf = int((front_line[0] - j*np.cos(front_line[1])) / np.sin(front_line[1]))
            yb = max(0, min(h-1, yb))
            yf = max(0, min(h-1, yf))
            if yb < yf:
                mask[yb:yf, j] = 1
    else:
        mask[:] = 1
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    edges  = cv2.Canny(masked, 50, 200, apertureSize=3)
    for thresh in [250, 180, 120]:
        for angle_range in [(0, 40), (130, 180)]:
            lines = cv2.HoughLines(edges, 1, np.pi/180/2, thresh, None,
                                   min_theta=angle_range[0]/180*np.pi,
                                   max_theta=angle_range[1]/180*np.pi)
            if lines is not None:
                return lines[0][0]
    return None


def _find_central_circle(gray, back_mid, front_mid, main_line):
    """Localizza il cerchio centrale con floodfill su Canny."""
    if back_mid is None or front_mid is None or main_line is None:
        return None, None, None, None

    edges = cv2.Canny(gray, 20, 100, apertureSize=3)
    fill  = cv2.dilate(edges, np.ones((7, 7)))
    h, w  = fill.shape
    mask  = np.zeros((h+2, w+2), np.uint8)

    bm  = np.array(back_mid)
    fm  = np.array(front_mid)
    ctr = (0.3*fm + 0.7*bm).astype(int)

    for seed_off in [-150, -100, -50, 0, 50, 100, 150]:
        root = (int(ctr[0]) + seed_off, int(ctr[1]))
        if 0 <= root[0] < w and 0 <= root[1] < h:
            if fill[root[1], root[0]] == 0:
                cv2.floodFill(fill, mask, root, 128)

    final = cv2.inRange(fill, 127, 129)
    final = cv2.dilate(final, np.ones((15, 15)))
    final = cv2.erode(final,  np.ones((10, 10)))

    cnts = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return None, None, None, None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, None, None, None

    left_c  = tuple(c[c[:,:,0].argmin()][0])
    right_c = tuple(c[c[:,:,0].argmax()][0])
    y_top   = int(c[c[:,:,1].argmin()][0][1])
    y_bot   = int(c[c[:,:,1].argmax()][0][1])

    rho, theta = main_line
    behind_c = [int((rho - y_top * np.sin(theta)) / np.cos(theta))  if abs(np.cos(theta))>1e-4 else int(bm[0]), y_top]
    front_c  = [int((rho - y_bot * np.sin(theta)) / np.cos(theta))  if abs(np.cos(theta))>1e-4 else int(bm[0]), y_bot]

    # sanity checks
    if left_c[0] == 0 or left_c[0] >= behind_c[0] - 10:  left_c  = None
    if right_c[0] >= w-1 or right_c[0] <= behind_c[0]+10: right_c = None

    return left_c, right_c, behind_c, front_c


def _detect_keypoints(frame):
    """
    Rileva i keypoints semantici del campo e restituisce
    (pixels_2d, points_3d) per il solver PnP.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Linee laterali
    back_line, front_line = _find_back_front_lines(gray)

    # 2. Linea mezzeria
    main_line = _find_main_line(gray, back_line, front_line)

    # 3. Punti intersezione linea mediana con laterali
    back_mid  = _intersect_polar(back_line,  main_line) if back_line  and main_line else None
    front_mid = _intersect_polar(front_line, main_line) if front_line and main_line else None

    # 4. Cerchio centrale
    left_c, right_c, behind_c, front_c = _find_central_circle(gray, back_mid, front_mid, main_line)

    # 5. Angoli del campo
    corner_bl = _intersect_polar(back_line,  _vertical_left(w))  if back_line  else None
    corner_br = _intersect_polar(back_line,  _vertical_right(w)) if back_line  else None
    corner_fl = _intersect_polar(front_line, _vertical_left(w))  if front_line else None
    corner_fr = _intersect_polar(front_line, _vertical_right(w)) if front_line else None

    # Costruisci lista 2D/3D
    associations = [
        ("right_circle",       right_c),
        ("left_circle",        left_c),
        ("behind_circle",      behind_c),
        ("front_circle",       front_c),
        ("back_middle_line",   back_mid),
        ("front_middle_line",  front_mid),
        ("corner_back_left",   corner_bl),
        ("corner_back_right",  corner_br),
        ("corner_front_left",  corner_fl),
        ("corner_front_right", corner_fr),
    ]

    pixels_2d = []
    points_3d = []
    for name, pt in associations:
        if pt is not None and _in_frame(pt, w, h):
            pixels_2d.append([float(pt[0]), float(pt[1])])
            points_3d.append(KP_WORLD[name])

    if len(pixels_2d) < 4:
        return None, None

    return np.array(pixels_2d, dtype=np.float32), np.array(points_3d, dtype=np.float64)


def _vertical_left(w):
    """Linea verticale al bordo sinistro dell'immagine in coord polari."""
    return np.array([0.0, np.pi/2])  # rho=0, theta=90°→ x=0

def _vertical_right(w):
    return np.array([float(w), np.pi/2])

def _in_frame(pt, w, h, margin=20):
    return margin <= pt[0] <= w-margin and margin <= pt[1] <= h-margin


# ── PnP camera pose solver ─────────────────────────────────────────────────────

def _solve_pnp(pixels_2d, points_3d, img_shape, guess_fx, guess_rvec, guess_tvec):
    """
    Risolve PnP e restituisce (K, rvec, tvec) oppure None se fallisce.
    """
    h, w = img_shape[:2]

    # Stima focale dal cerchio se possibile
    fx = guess_fx
    rc_idx = None; lc_idx = None
    for i, p3 in enumerate(points_3d):
        if np.allclose(p3, KP_WORLD["right_circle"]): rc_idx = i
        if np.allclose(p3, KP_WORLD["left_circle"]):  lc_idx = i
    if rc_idx is not None and lc_idx is not None:
        dx_px = abs(pixels_2d[rc_idx][0] - pixels_2d[lc_idx][0])
        dx_m  = abs(KP_WORLD["right_circle"][0] - KP_WORLD["left_circle"][0])
        # dist camera ≈ 77m
        fx_est = dx_px * 77.0 / dx_m
        if 500 < fx_est < 5000:
            fx = fx_est

    K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)

    try:
        ret, rvec, tvec = cv2.solvePnP(
            points_3d, pixels_2d, K, None,
            rvec=guess_rvec, tvec=guess_tvec,
            useExtrinsicGuess=(guess_rvec is not None),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except Exception:
        return None, None, None, None

    if not ret or np.any(np.isnan(rvec)):
        return None, None, None, None

    # Sanity: camera deve essere tra 40 e 100m dal centro campo
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec
    dist = float(np.linalg.norm(cam_pos))
    if not (30.0 < dist < 120.0):
        return None, None, None, None

    return K, rvec, tvec, fx


# ── Omografia 2D fallback (Hough) ─────────────────────────────────────────────

FIELD_CORNERS_3D = np.array([
    [-FIELD_W/2, -FIELD_H/2],
    [ FIELD_W/2, -FIELD_H/2],
    [ FIELD_W/2,  FIELD_H/2],
    [-FIELD_W/2,  FIELD_H/2],
], dtype=np.float32)


def _hough_homography(frame):
    """Omografia 2D da linee Hough — fallback se PnP non converge."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    white = cv2.bitwise_and(white, mask)
    white = cv2.morphologyEx(white, cv2.MORPH_DILATE, np.ones((3, 3)))

    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                            minLineLength=40, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None, "GreenBBox"

    pts = np.array([[x1, y1] for x1, y1, x2, y2 in lines[:, 0]] +
                   [[x2, y2] for x1, y1, x2, y2 in lines[:, 0]], dtype=np.float32)

    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) < 4:
        return None, "GreenBBox"

    # Seleziona 4 angoli dal convex hull
    cx, cy = hull.mean(axis=0)
    corners = [None, None, None, None]  # TL, TR, BR, BL
    best    = [np.inf]*4
    for p in hull:
        dx, dy = p[0]-cx, p[1]-cy
        if   dx < 0 and dy < 0: idx = 0
        elif dx > 0 and dy < 0: idx = 1
        elif dx > 0 and dy > 0: idx = 2
        else:                    idx = 3
        d = dx*dx + dy*dy
        if d < best[idx]: continue
        best[idx] = d; corners[idx] = p

    if any(c is None for c in corners):
        return None, "GreenBBox"

    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0], [MAP_W-2*MAP_MARGIN, 0],
        [MAP_W-2*MAP_MARGIN, MAP_H-2*MAP_MARGIN], [0, MAP_H-2*MAP_MARGIN]
    ], dtype=np.float32) + MAP_MARGIN

    H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask_h.sum() < 4:
        return None, "GreenBBox"
    return H, "Hough"


def _green_bbox_homography(frame):
    """Fallback finale: bounding box del verde."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    src = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    dst = np.array([
        [MAP_MARGIN, MAP_MARGIN],
        [MAP_W-MAP_MARGIN, MAP_MARGIN],
        [MAP_W-MAP_MARGIN, MAP_H-MAP_MARGIN],
        [MAP_MARGIN, MAP_H-MAP_MARGIN]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


# ── HomographyMapper principale ───────────────────────────────────────────────

class HomographyMapper:
    def __init__(self):
        # Stato PnP
        self.K            = None
        self._rvec        = None
        self._tvec        = None
        self._guess_fx    = 1800.0
        self._pnp_ok      = False
        # Stato omografia 2D
        self.H            = None
        self.H_inv        = None
        self._cal_method  = "GreenBBox"
        # Calibrazione periodica
        self._cal_every   = 25
        self._frame_cnt   = 0
        # Giocatori e palla per minimap
        self._players     = {}     # tid → (team_id, (x_m, z_m))
        self._ball_map    = None
        # Smoothing omografia
        self._H_buf       = deque(maxlen=5)
        # Scala pixel/metro per StatsTracker
        self.px_per_m     = None

    # ── Aggiornamenti frame ────────────────────────────────────────────────────

    def update_frame(self, frame):
        self._frame_cnt += 1
        if self._frame_cnt % self._cal_every != 1:
            return
        self._calibrate(frame)

    def _calibrate(self, frame):
        """Prova PnP → Hough → GreenBBox in cascata."""
        h, w = frame.shape[:2]

        # ── Tentativo 1: PnP ────────────────────────────────────────────────
        pixels_2d, points_3d = _detect_keypoints(frame)
        if pixels_2d is not None and len(pixels_2d) >= 4:
            K, rvec, tvec, fx = _solve_pnp(
                pixels_2d, points_3d, frame.shape,
                self._guess_fx, self._rvec, self._tvec
            )
            if K is not None:
                self.K          = K
                self._rvec      = rvec
                self._tvec      = tvec
                self._guess_fx  = fx
                self._pnp_ok    = True
                self._cal_method = "PnP"
                self._update_H_from_pnp(w, h)
                self._update_scale(w, h)
                return

        # ── Tentativo 2: Hough ──────────────────────────────────────────────
        self._pnp_ok = False
        H_hough, method = _hough_homography(frame)
        if H_hough is not None:
            self._H_buf.append(H_hough)
            # Media delle ultime H per smoothing
            H_avg = np.mean(self._H_buf, axis=0)
            self.H   = H_avg
            try:
                self.H_inv = np.linalg.inv(H_avg)
            except Exception:
                self.H_inv = None
            self._cal_method = method
            self._update_scale_from_H(w, h)
            return

        # ── Tentativo 3: GreenBBox ──────────────────────────────────────────
        H_gb = _green_bbox_homography(frame)
        if H_gb is not None:
            self.H   = H_gb
            try:
                self.H_inv = np.linalg.inv(H_gb)
            except Exception:
                self.H_inv = None
            self._cal_method = "GreenBBox"
            self._update_scale_from_H(w, h)

    def _update_H_from_pnp(self, w, h):
        """Costruisce omografia pixel→minimappa dal risultato PnP."""
        try:
            # 4 angoli campo 3D → pixel immagine
            corners_3d = np.array([
                [-FIELD_W/2, 0, -FIELD_H/2],
                [ FIELD_W/2, 0, -FIELD_H/2],
                [ FIELD_W/2, 0,  FIELD_H/2],
                [-FIELD_W/2, 0,  FIELD_H/2],
            ], dtype=np.float64)
            corners_2d, _ = cv2.projectPoints(
                corners_3d, self._rvec, self._tvec, self.K, None)
            corners_2d = corners_2d.reshape(-1, 2).astype(np.float32)

            dst = np.array([
                [MAP_MARGIN,         MAP_MARGIN],
                [MAP_W-MAP_MARGIN,   MAP_MARGIN],
                [MAP_W-MAP_MARGIN,   MAP_H-MAP_MARGIN],
                [MAP_MARGIN,         MAP_H-MAP_MARGIN],
            ], dtype=np.float32)

            H, mask = cv2.findHomography(corners_2d, dst, cv2.RANSAC, 5.0)
            if H is not None and mask is not None and mask.sum() >= 3:
                self._H_buf.append(H)
                self.H   = np.mean(self._H_buf, axis=0)
                self.H_inv = np.linalg.inv(self.H)
        except Exception:
            pass

    def _update_scale(self, w, h):
        """Calcola px_per_m dal modello PnP."""
        if self.K is None or self._rvec is None:
            return
        try:
            p0, _ = cv2.projectPoints(
                np.array([[0.0, 0.0, 0.0]]), self._rvec, self._tvec, self.K, None)
            p1, _ = cv2.projectPoints(
                np.array([[1.0, 0.0, 0.0]]), self._rvec, self._tvec, self.K, None)
            self.px_per_m = float(np.linalg.norm(
                p1.reshape(2) - p0.reshape(2)))
        except Exception:
            pass

    def _update_scale_from_H(self, w, h):
        """Stima px_per_m dall'omografia 2D."""
        if self.H is None:
            return
        try:
            fw = MAP_W - 2*MAP_MARGIN
            self.px_per_m = fw / FIELD_W * (w / (MAP_W or 1))
        except Exception:
            pass

    # ── Pixel immagine → coordinate campo (metri) ─────────────────────────────

    def pixel_to_field(self, pixel_pt):
        """Converte pixel → (x_m, z_m) nel sistema campo."""
        px, py = float(pixel_pt[0]), float(pixel_pt[1])

        # Metodo PnP: ray casting sul piano y=0
        if self._pnp_ok and self.K is not None and self._rvec is not None:
            try:
                K_inv = np.linalg.inv(self.K)
                R, _  = cv2.Rodrigues(self._rvec)
                t     = self._tvec.reshape(3)
                ray_cam = K_inv @ np.array([px, py, 1.0])
                # Intersezione con piano y=0: R*X + t = lambda*ray_cam
                # y=0 → (R[1,:]*X) + t[1] = 0
                denom = R[1] @ ray_cam
                if abs(denom) < 1e-6:
                    raise ValueError("ray parallelo al piano")
                lam = -( R[1] @ (-R.T @ t) + 0 ) / denom  # semplificato
                # Risolvo con formula completa
                # (R^T)(lambda*ray_cam - t) = X_world
                X_cam = lam * ray_cam
                X_world = R.T @ (X_cam - t)
                x_m, z_m = float(X_world[0]), float(X_world[2])
                if -60 < x_m < 60 and -40 < z_m < 40:
                    return (x_m, z_m)
            except Exception:
                pass

        # Metodo omografia 2D
        if self.H is not None:
            try:
                pt = cv2.perspectiveTransform(
                    np.array([[[px, py]]], dtype=np.float32), self.H)[0][0]
                # Converti coord mappa → metri
                u, v = float(pt[0]), float(pt[1])
                fw = MAP_W - 2*MAP_MARGIN
                fh = MAP_H - 2*MAP_MARGIN
                x_m = (u - MAP_MARGIN) / fw * FIELD_W - FIELD_W/2
                z_m = (v - MAP_MARGIN) / fh * FIELD_H - FIELD_H/2
                if -60 < x_m < 60 and -40 < z_m < 40:
                    return (x_m, z_m)
            except Exception:
                pass

        return None

    # ── Aggiornamenti giocatori / palla ───────────────────────────────────────

    def update_players(self, field_positions):
        self._players = field_positions

    def update_ball(self, pixel_center):
        self._ball_map = pixel_center

    # ── Rendering minimappa ───────────────────────────────────────────────────

    def render_minimap(self):
        img = _BASE_MINIMAP.copy()

        # Label metodo calibrazione
        cv2.putText(img, f"MINIMAP [{self._cal_method}]",
                    (4, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (180, 180, 180), 1, cv2.LINE_AA)

        # Giocatori
        T0_BGR = ( 55, 135, 255)
        T1_BGR = (255,  65,  65)
        REF_BGR= (185, 145, 255)
        for tid, (team_id, fp) in self._players.items():
            if fp is None:
                continue
            x_m, z_m = fp
            u, v = _field_to_map(x_m, z_m)
            if 0 <= u < MAP_W and 0 <= v < MAP_H:
                col = T0_BGR if team_id == 0 else (T1_BGR if team_id == 1 else REF_BGR)
                cv2.circle(img, (u, v), 5, col, -1)
                cv2.circle(img, (u, v), 5, (255,255,255), 1)

        # Palla
        if self._ball_map is not None:
            fp = self.pixel_to_field(self._ball_map)
            if fp:
                u, v = _field_to_map(*fp)
                if 0 <= u < MAP_W and 0 <= v < MAP_H:
                    cv2.circle(img, (u, v), 4, _DOT_BALL, -1)

        return img