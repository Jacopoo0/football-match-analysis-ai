import cv2
from ultralytics import YOLO
from minimap import (
    create_minimap,
    convert_to_minimap_coords,
    draw_player_on_minimap,
    draw_player_trail
)

video_path = "data/raw/partita.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

track_history = {}


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


cv2.namedWindow("Football Analysis Dashboard", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_height, frame_width = frame.shape[:2]
    result = model.track(frame, persist=True)[0]
    minimap = create_minimap()

    active_ids = []

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            active_ids.append(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            map_x, map_y = convert_to_minimap_coords(
                foot_x,
                foot_y,
                frame_width,
                frame_height
            )

            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((map_x, map_y))

            if len(track_history[track_id]) > 20:
                track_history[track_id] = track_history[track_id][-20:]

            draw_player_trail(minimap, track_history[track_id])
            draw_player_on_minimap(minimap, map_x, map_y, track_id)

    frame_small = resize_with_aspect_ratio(frame, width=700)
    minimap_small = resize_with_aspect_ratio(minimap, height=frame_small.shape[0])

    cv2.putText(frame_small, "VIDEO TRACKING", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(minimap_small, "MINIMAP LIVE", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    dashboard = cv2.hconcat([frame_small, minimap_small])

    cv2.imshow("Football Analysis Dashboard", dashboard)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()