from pathlib import Path
import cv2
from ultralytics import YOLO
from team_classifier import TeamClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
OUTPUT_JSON = BASE_DIR / "team_colors.json"
SAMPLES_PER_TEAM = 3


classifier = TeamClassifier()
selected = []
frame = None
base_frame = None
person_boxes = []


def detect_players(img, model):
    results = model(img, classes=[0], verbose=True)[0]
    boxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        area = max(1, (x2 - x1) * (y2 - y1))

        if conf < 0.25:
            continue
        if area < 500:
            continue

        boxes.append((x1, y1, x2, y2, conf))

    return boxes


def team_color(team_id):
    if team_id == 0:
        return (255, 0, 0)
    if team_id == 1:
        return (0, 0, 255)
    return (0, 255, 255)


def current_team_to_pick():
    if len(selected) < SAMPLES_PER_TEAM:
        return 0
    return 1


def redraw():
    global frame
    frame = base_frame.copy()

    for (x1, y1, x2, y2, conf) in person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for item in selected:
        x1, y1, x2, y2 = item["bbox"]
        t = item["team_id"]
        color = team_color(t)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            f"T{t}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    t0 = sum(1 for x in selected if x["team_id"] == 0)
    t1 = sum(1 for x in selected if x["team_id"] == 1)

    if len(selected) < 2 * SAMPLES_PER_TEAM:
        next_team = current_team_to_pick()
        msg = f"Clicca TEAM {next_team} ({t0}/{SAMPLES_PER_TEAM} - {t1}/{SAMPLES_PER_TEAM})"
    else:
        msg = "Premi S per salvare, R per ricominciare, Q per uscire"

    cv2.putText(
        frame,
        msg,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


def find_box_for_point(x, y):
    containing = []

    for (x1, y1, x2, y2, conf) in person_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            containing.append((area, (x1, y1, x2, y2)))

    if containing:
        containing.sort(key=lambda z: z[0])
        return containing[0][1]

    best_box = None
    best_dist = 10**9

    for (x1, y1, x2, y2, conf) in person_boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = (cx - x) ** 2 + (cy - y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_box = (x1, y1, x2, y2)

    return best_box


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(selected) >= 2 * SAMPLES_PER_TEAM:
        return

    bbox = find_box_for_point(x, y)
    if bbox is None:
        return

    feature = classifier.extract_jersey_feature(base_frame, bbox)
    if feature is None:
        return

    team_id = current_team_to_pick()
    classifier.add_sample(team_id, feature)
    selected.append({"team_id": team_id, "bbox": bbox})

    print(f"Aggiunto sample TEAM {team_id}: bbox={bbox}")
    redraw()


def main():
    global frame, base_frame, person_boxes, classifier, selected

    print("VIDEO_PATH =", VIDEO_PATH)
    print("MODEL_PATH =", MODEL_PATH)
    print("OUTPUT_JSON =", OUTPUT_JSON)

    if not VIDEO_PATH.exists():
        print(f"Video non trovato: {VIDEO_PATH}")
        return

    if not MODEL_PATH.exists():
        print(f"Modello non trovato: {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, first_frame = cap.read()
    cap.release()

    if not ok:
        print(f"Errore: impossibile leggere il video {VIDEO_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    base_frame = first_frame.copy()
    person_boxes = detect_players(base_frame, model)

    if len(person_boxes) == 0:
        print("Nessun giocatore rilevato nel primo frame")
        return

    redraw()

    cv2.namedWindow("Select Team Colors")
    cv2.setMouseCallback("Select Team Colors", on_mouse)

    while True:
        cv2.imshow("Select Team Colors", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            classifier = TeamClassifier()
            selected = []
            redraw()
            print("Reset completato")

        elif key == ord("s"):
            t0 = len(classifier.samples[0])
            t1 = len(classifier.samples[1])

            if t0 == SAMPLES_PER_TEAM and t1 == SAMPLES_PER_TEAM:
                classifier.save_samples(str(OUTPUT_JSON))
                print(f"Samples salvati in {OUTPUT_JSON}")
                print(f"TEAM 0 samples: {t0}")
                print(f"TEAM 1 samples: {t1}")
                break
            else:
                print("Devi selezionare 3 giocatori per TEAM 0 e 3 giocatori per TEAM 1")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()