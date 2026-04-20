import cv2
from ultralytics import YOLO
from minimap import create_minimap, draw_player_on_minimap
from homography import build_homography_matrices, get_current_homography, project_point
from team_classifier import TeamClassifier
from collections import defaultdict, deque, Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
TEAM_JSON = BASE_DIR / "team_colors.json"

CONF_THRESH = 0.35
IMG_SIZE = 640
TEAM_HISTORY_LEN = 12


def resize_with_aspect_ratio(image, width=None, height=None):
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        scale = width / w
        new_width = width
        new_height = int(h * scale)
    else:
        scale = height / h
        new_height = height
        new_width = int(w * scale)

    return cv2.resize(image, (new_width, new_height))


def draw_info(frame, current_sec, segment):
    cv2.putText(
        frame,
        f"TIME: {current_sec:.1f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"SEGMENT: {segment['start_sec']:.0f}-{segment['end_sec']:.0f}s",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SEMI-AUTO TEAM COLORS",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


def get_team_color_and_label(team_id):
    if team_id == 0:
        return (255, 0, 0), "TEAM 0"
    if team_id == 1:
        return (0, 0, 255), "TEAM 1"
    return (0, 255, 255), "UNK"


def main():
    print("VIDEO_PATH =", VIDEO_PATH)
    print("MODEL_PATH =", MODEL_PATH)
    print("TEAM_JSON =", TEAM_JSON)

    if not VIDEO_PATH.exists():
        print(f"Video non trovato: {VIDEO_PATH}")
        return

    if not MODEL_PATH.exists():
        print(f"Modello non trovato: {MODEL_PATH}")
        return

    if not TEAM_JSON.exists():
        print(f"File team colors non trovato: {TEAM_JSON}")
        print("Esegui prima: python .\\src\\select_team_colors.py")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Impossibile aprire il video: {VIDEO_PATH}")
        return

    homography_segments = build_homography_matrices()

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    team_history = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))

    cv2.namedWindow("Football Analysis Dashboard", cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_height, frame_width = frame.shape[:2]
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        current_segment = get_current_homography(current_sec, homography_segments)
        H = current_segment["matrix"]

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONF_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )

        result = results[0]
        minimap = create_minimap()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()

            if result.boxes.cls is not None:
                class_ids = result.boxes.cls.int().cpu().tolist()
            else:
                class_ids = [0] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2, y2)

                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                frame_area = frame_width * frame_height

                if box_area > frame_area * 0.12:
                    continue

                if box_w / max(box_h, 1) > 1.2:
                    continue

                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                team_id_raw, team_conf = team_classifier.classify_player(frame, bbox)

                if team_id_raw != -1:
                    team_history[int(track_id)].append(team_id_raw)

                if len(team_history[int(track_id)]) >= 3:
                    team_id = Counter(team_history[int(track_id)]).most_common(1)[0][0]
                else:
                    team_id = team_id_raw

                draw_color, draw_label = get_team_color_and_label(team_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.circle(frame, (foot_x, foot_y), 4, draw_color, -1)

                cv2.putText(
                    frame,
                    f"ID {track_id} {draw_label} {team_conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    draw_color,
                    2
                )

                map_x, map_y = project_point(foot_x, foot_y, H)

                if 0 <= map_x < 500 and 0 <= map_y < 320:
                    draw_player_on_minimap(minimap, map_x, map_y, int(track_id), draw_color)

        draw_info(frame, current_sec, current_segment)

        cv2.putText(
            minimap,
            "BLUE/RED = TEAMS | YELLOW = UNK",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        frame_small = resize_with_aspect_ratio(frame, width=700)
        minimap_small = resize_with_aspect_ratio(minimap, height=frame_small.shape[0])

        dashboard = cv2.hconcat([frame_small, minimap_small])

        cv2.imshow("Football Analysis Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()