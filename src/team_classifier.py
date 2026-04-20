import json
import cv2
import numpy as np


class TeamClassifier:
    def __init__(self):
        self.samples = {0: [], 1: []}
        self.unknown_margin = 0.10

    def _clip_bbox(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        return x1, y1, x2, y2

    def extract_jersey_feature(self, frame, bbox):
        x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        h, w = crop.shape[:2]

        y_start = int(0.10 * h)
        y_end = max(int(0.55 * h), y_start + 1)
        x_start = int(0.15 * w)
        x_end = max(int(0.85 * w), x_start + 1)

        jersey = crop[y_start:y_end, x_start:x_end]
        if jersey.size == 0:
            jersey = crop

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        green_mask = cv2.inRange(hsv, (35, 30, 20), (95, 255, 255))
        non_green = cv2.bitwise_not(green_mask)

        informative = np.where((s > 25) | (v > 160), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(non_green, informative)

        if cv2.countNonZero(mask) < 0.02 * mask.size:
            mask = non_green

        if cv2.countNonZero(mask) < 0.02 * mask.size:
            mask = np.full(mask.shape, 255, dtype=np.uint8)

        hist_h = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()

        if hist_h.sum() > 0:
            hist_h = hist_h / hist_h.sum()

        if hist_s.sum() > 0:
            hist_s = hist_s / hist_s.sum()

        mean_bgr = np.array(cv2.mean(jersey, mask=mask)[:3], dtype=np.float32) / 255.0

        pixels = jersey[mask > 0]
        if len(pixels) > 0:
            std_bgr = pixels.std(axis=0).astype(np.float32) / 255.0
        else:
            std_bgr = np.zeros(3, dtype=np.float32)

        feature = np.concatenate([hist_h, hist_s, mean_bgr, std_bgr]).astype(np.float32)
        return feature

    def add_sample(self, team_id, feature):
        if feature is not None:
            self.samples[int(team_id)].append(feature)

    def save_samples(self, path):
        data = {
            "team_0": [s.tolist() for s in self.samples[0]],
            "team_1": [s.tolist() for s in self.samples[1]],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_samples(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples[0] = [np.array(s, dtype=np.float32) for s in data.get("team_0", [])]
        self.samples[1] = [np.array(s, dtype=np.float32) for s in data.get("team_1", [])]

    def _team_distance(self, feature, team_id):
        team_samples = self.samples[team_id]
        if not team_samples:
            return 1e9

        dists = [float(np.linalg.norm(feature - s)) for s in team_samples]
        dists.sort()

        k = min(2, len(dists))
        return 0.7 * dists[0] + 0.3 * np.mean(dists[:k])

    def classify_feature(self, feature):
        if feature is None:
            return -1, 0.0

        d0 = self._team_distance(feature, 0)
        d1 = self._team_distance(feature, 1)

        if not np.isfinite(d0) or not np.isfinite(d1):
            return -1, 0.0

        if d0 <= d1:
            best_team, best_dist, other_dist = 0, d0, d1
        else:
            best_team, best_dist, other_dist = 1, d1, d0

        margin = (other_dist - best_dist) / max(other_dist, 1e-6)
        confidence = float(np.clip(margin, 0.0, 1.0))

        if margin < self.unknown_margin:
            return -1, confidence

        return best_team, confidence

    def classify_player(self, frame, bbox):
        feature = self.extract_jersey_feature(frame, bbox)
        return self.classify_feature(feature)