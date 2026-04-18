import cv2
from ultralytics import YOLO

video_path = "data/raw/partita.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True)
    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f"ID {track_id} -> box=({x1}, {y1}, {x2}, {y2}) foot=({foot_x}, {foot_y})")

    cv2.imshow("Tracking con foot point", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()