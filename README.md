

````markdown name=README.md


# ⚽ MiniMappaFootBall
**Real-time football video analysis — player tracking, team classification, match statistics & tactical minimap**

MiniMappaFootBall is a computer vision pipeline that processes football match footage in real time — detecting players and the ball with a custom YOLOv8 model, assigning persistent multi-object track IDs via BoTSORT with camera motion compensation, classifying each player into their team by jersey colour, and rendering a live DAZN-style analytics overlay with a calibrated tactical minimap.

</div>

---

📸 **Output Preview**

![output preview](https://github.com/user-attachments/assets/5441e3b9-ddff-4a95-ab93-431987e5c236)
<img width="1918" height="781" alt="Screenshot 2026-05-06 195221" src="https://github.com/user-attachments/assets/ad7a344f-4531-42ae-a3e2-e6743d09b84c" />

---

## ✨ Features

| Module | Description |
|---|---|
| 🔍 Detection | YOLOv8 with custom `soccana_best.pt` — detects Player, Ball, Referee per frame |
| 🎯 Tracking | BoTSORT with camera motion compensation (`cmc_method=sof`) — stable IDs across cuts and occlusions |
| 👕 Team Classifier | Dominant jersey colour extracted in LAB space via KMeans; 30-frame history buffer stabilises noisy frames |
| 📊 Stats Engine | Per-team: possession %, passes, avg/max speed (km/h), distance (km) — computed in real time |
| 🗺️ Tactical Minimap | Automatic homography from green-field contour detection; recalibrates every 30 frames for pan/zoom |
| 🎨 DAZN-style UI | Dark professional panel built with Pillow + Inter font; split possession bar, colour-coded cards, progress strip |
| 🖼️ Preprocessing | CLAHE contrast enhancement + sharpening kernel applied before each inference |
| ⚡ Threaded Pipeline | `InferenceWorker` thread decouples inference from video I/O; no frame dropping in the main loop |

---

## 📁 Project Structure

```
MiniMappaFootBall/
│
├── data/
│   └── raw/
│       └── input_vid.mp4               ← source footage
│
├── models/
│   ├── soccana_best.pt                 ← custom YOLO weights (football)
│   └── osnet_x0_25_msmt17.pt           ← ReID weights for BoTSORT
│
├── src/
│   ├── main.py                         ← pipeline entry point & canvas rendering
│   ├── team_classifier.py              ← LAB-space jersey colour classification
│   ├── stats_tracker.py                ← possession, passes, speed, distance
│   ├── homography.py                   ← bird's-eye minimap via homography
│   └── select_team_colors.py           ← interactive colour sampler (run once)
│
├── .font_cache/
│   ├── Inter-Regular.otf
│   └── Inter-Bold.otf
│
├── team_colors.json                    ← LAB centroids per team (generated)
├── output_football_analysis.mp4        ← analysis output (generated)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 · Prerequisites
- Python 3.12  
- NVIDIA GPU + CUDA 12.1 (tested: RTX 4060 Laptop)  
- PyTorch 2.5.1 with CUDA

### 2 · Install
```bash
git clone https://github.com/YOUR_USERNAME/MiniMappaFootBall.git
cd MiniMappaFootBall

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 · Calibrate Team Colours (run once per video)
```bash
python src/select_team_colors.py
```
Draw rectangles over player jerseys → TAB to switch team → ENTER to save `team_colors.json`.

### 4 · Run Analysis
```bash
python src/main.py
```
Output is written to `output_football_analysis.mp4`.

---

## ⚙️ Configuration

Edit the constants block at the top of `src/main.py`:

```python
MAX_SECONDS  = 60      # seconds to process  (None = full video)
PLAYER_CONF  = 0.18    # YOLO confidence threshold
INFER_SIZE   = 640     # inference resolution — higher → more accurate, slower
SAVE_VIDEO   = True    # write output .mp4
SHOW_PREVIEW = False   # display live OpenCV window
FRAME_W      = 960     # output frame width
FRAME_H      = 540     # output frame height
PANEL_W      = 360     # stats panel width
```

---

## 🏗️ Pipeline Architecture

```
┌──────────────┐
│  Video Frame │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│  preprocess_frame() │  CLAHE + sharpening kernel
└──────┬──────────────┘
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
┌─────────────────┐                    ┌──────────────────┐
│  YOLO (players) │                    │   YOLO (ball)    │
│  conf=0.18      │                    │   conf=0.05      │
└──────┬──────────┘                    └────────┬─────────┘
       │                                        │
       ▼                                        ▼
┌──────────────┐                      ┌──────────────────┐
│   BoTSORT    │  CMC: sof            │   ball_center    │
│   track IDs  │  track_buffer=120    │   (pixel coords) │
└──────┬───────┘                      └────────┬─────────┘
       │                                        │
       ▼                                        │
┌─────────────────┐                             │
│ TeamClassifier  │                             │
│ LAB → team 0/1  │                             │
│ history=30f     │                             │
└──────┬──────────┘                             │
       │                                        │
       └──────────────┬─────────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │    StatsTracker     │
           │  possession         │
           │  passes (≥6f streak)│
           │  speed / distance   │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │  HomographyMapper   │
           │  green mask → H     │
           │  recal every 30f    │
           │  render_minimap()   │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │  build_panel()      │
           │  build_canvas()     │
           └──────────┬──────────┘
                      │
                      ▼
           output_football_analysis.mp4
```

---

## 📊 Stats Reference

| Metric | Computation |
|---|---|
| Possession | Team of the player closest to the ball (threshold: 90 px) |
| Recent possession | Rolling 5-second window |
| Pass | Team change event after ≥ 6 consecutive frames of possession by the previous team |
| Speed | Δpx × 0.058 m/px × fps × 3.6 → km/h; per-player rolling window (1 s) |
| Distance | Cumulative displacement per player, filtered to 0.5 – 60 px / frame |
| Max speed | Per-team rolling maximum, physically capped at 60 km/h |

---

## 🗺️ Minimap & Homography

The minimap is a 300 × 194 px bird's-eye view of a standard 105 × 68 m pitch, calibrated automatically each run:

- HSV green mask → isolates the pitch surface
- Morphological cleaning (CLOSE + OPEN, 9 × 9 ellipse) → removes crowd / ad-board noise
- approxPolyDP on the largest green contour → extracts 4 field corners; falls back to bounding rect (± 3% margin) when the polygon has ≠ 4 vertices
- `cv2.findHomography` (RANSAC) → pixel space to metric space mapping
- Recalibration every 30 frames → handles camera pan / zoom / cut

Player foot-point (cx, cy) is projected through H and rendered as a coloured dot with a white ring. The ball is rendered as a yellow dot with a glow.

---

## 🛣️ Roadmap

- Heatmaps — per-player and per-team spatial density overlays  
- Formation detection — automatic recognition of 4-4-2, 4-3-3, etc.  
- Event detection — shot on goal, corner kick, throw-in  
- CSV / JSON export — per-frame structured statistics  
- Web dashboard — Angular frontend + Django REST backend  
- Multi-camera support — cross-view homography stitching  
- 1080p / 4K input — optimised inference for broadcast quality video

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss significant changes.

```bash
# Fork → branch → commit → pull request
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Built with** &nbsp;  
YOLOv8 · BoxMOT · Supervision · OpenCV · Pillow · scikit-learn

If you find this project useful, consider leaving a ⭐

</div>
```
