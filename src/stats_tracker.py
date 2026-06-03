from collections import defaultdict, deque
import numpy as np


METERS_PER_PIXEL = 0.058
MAX_SPEED_KMH    = 36.0   # cap fisiologico: nessun calciatore supera i 36 km/h


class StatsTracker:
    def __init__(self, fps=25.0):
        self.fps              = fps
        self.possession       = {0: 0, 1: 0}
        self.dist_px          = defaultdict(float)
        self.team_of          = {}
        self.last_pos         = {}
        self.poss_history     = deque(maxlen=int(fps * 5))
        self.speed_hist       = defaultdict(lambda: deque(maxlen=int(fps)))
        self.passes           = {0: 0, 1: 0}
        self._ball_positions  = deque(maxlen=int(fps * 2))
        self._ball_owner_team = -1
        self._owner_streak    = 0
        self._pass_cooldown   = 0
        self._prev_free_frames = 0
        self._debug_counter   = 0
        # Scala pixel/metro: aggiornata da HomographyMapper se calibrato
        self._px_per_m        = None

        self.POSSESS_DIST    = 90
        self.PASS_MIN_STREAK = 8    # alzato da 6: riduce falsi positivi
        self.PASS_COOLDOWN   = 20   # alzato da 15: evita duplicati su tackle

    # ── Calibrazione scala dal campo reale ────────────────────────────────────
    def set_scale(self, px_per_m: float):
        """Riceve pixel/metro da HomographyMapper quando disponibile."""
        self._px_per_m = px_per_m

    def _px_to_m(self, px: float) -> float:
        """Converte pixel in metri usando scala reale o costante fallback."""
        if self._px_per_m and self._px_per_m > 0:
            return px / self._px_per_m
        return px * METERS_PER_PIXEL

    # ── Update per-frame ──────────────────────────────────────────────────────
    def update(self, ball_center, players):
        self._debug_counter += 1

        if self._pass_cooldown > 0:
            self._pass_cooldown -= 1

        if ball_center:
            self._ball_positions.append(ball_center)

        closest_team = -1
        closest_dist = float('inf')

        if ball_center and players:
            bx, by = ball_center
            for cx, cy, team_id, tid in players:
                if team_id not in (0, 1):
                    continue
                d = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
                if d < closest_dist:
                    closest_dist = d
                    closest_team = team_id

        in_possession = (closest_dist < self.POSSESS_DIST and closest_team in (0, 1))

        # ── Possesso ──────────────────────────────────────────────────────────
        if in_possession:
            self.possession[closest_team] += 1
            self.poss_history.append(closest_team)
        else:
            self.poss_history.append(-1)

        # ── Passaggi ──────────────────────────────────────────────────────────
        if in_possession:
            if closest_team == self._ball_owner_team:
                self._owner_streak += 1
            else:
                if (self._ball_owner_team in (0, 1)
                        and self._owner_streak >= self.PASS_MIN_STREAK
                        and self._pass_cooldown == 0):
                    self.passes[self._ball_owner_team] += 1
                    self._pass_cooldown = self.PASS_COOLDOWN
                self._ball_owner_team = closest_team
                self._owner_streak    = 1
            self._prev_free_frames = 0
        else:
            self._prev_free_frames += 1
            if self._prev_free_frames > 20:
                self._owner_streak = 0

        # ── Distanza + velocità ───────────────────────────────────────────────
        for cx, cy, team_id, tid in players:
            if tid == -1:
                continue
            if team_id in (0, 1):
                self.team_of[tid] = team_id
            if tid in self.last_pos:
                lx, ly = self.last_pos[tid]
                d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if 0.5 < d < 60:
                    self.dist_px[tid] += d
                    kmh = self._px_to_m(d) * self.fps * 3.6
                    # Cap fisiologico: elimina spike da ID instabili
                    if kmh <= MAX_SPEED_KMH:
                        self.speed_hist[tid].append(kmh)
            self.last_pos[tid] = (cx, cy)

    # ── Getters ───────────────────────────────────────────────────────────────
    def possession_pct(self):
        base = self.possession[0] + self.possession[1]
        if base == 0:
            return 50.0, 50.0
        return (
            round(self.possession[0] / base * 100, 1),
            round(self.possession[1] / base * 100, 1)
        )

    def recent_possession(self):
        h  = list(self.poss_history)
        t0, t1 = h.count(0), h.count(1)
        base = t0 + t1
        if base == 0:
            return 50.0, 50.0
        return round(t0 / base * 100, 1), round(t1 / base * 100, 1)

    def distance_meters(self):
        d = {0: 0.0, 1: 0.0}
        for tid, px in self.dist_px.items():
            t = self.team_of.get(tid, -1)
            if t in d:
                d[t] += self._px_to_m(px)
        return d[0], d[1]

    def avg_speed_kmh(self):
        speeds = {0: [], 1: []}
        for tid, hist in self.speed_hist.items():
            t = self.team_of.get(tid, -1)
            if t in speeds and len(hist) >= 3:
                speeds[t].append(float(np.mean(list(hist)[-10:])))
        s0 = round(float(np.mean(speeds[0])), 1) if speeds[0] else 0.0
        s1 = round(float(np.mean(speeds[1])), 1) if speeds[1] else 0.0
        return s0, s1

    def max_speed_kmh(self):
        maxs = {0: 0.0, 1: 0.0}
        for tid, hist in self.speed_hist.items():
            t = self.team_of.get(tid, -1)
            if t in maxs and hist:
                m = float(max(hist))
                if m > maxs[t]:
                    maxs[t] = m
        return round(maxs[0], 1), round(maxs[1], 1)