import cv2
import numpy as np

MINIMAP_WIDTH = 500
MINIMAP_HEIGHT = 320
PADDING = 20

def create_minimap():
    minimap = np.zeros((MINIMAP_HEIGHT, MINIMAP_WIDTH, 3), dtype=np.uint8)

    minimap[:] = (40, 120, 40)

    cv2.rectangle(
        minimap,
        (PADDING, PADDING),
        (MINIMAP_WIDTH - PADDING, MINIMAP_HEIGHT - PADDING),
        (255, 255, 255),
        2
    )

    cv2.line(
        minimap,
        (MINIMAP_WIDTH // 2, PADDING),
        (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT - PADDING),
        (255, 255, 255),
        2
    )

    cv2.circle(
        minimap,
        (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT // 2),
        40,
        (255, 255, 255),
        2
    )

    return minimap


def convert_to_minimap_coords(foot_x, foot_y, frame_width, frame_height):
    map_x = int((foot_x / frame_width) * (MINIMAP_WIDTH - 2 * PADDING) + PADDING)
    map_y = int((foot_y / frame_height) * (MINIMAP_HEIGHT - 2 * PADDING) + PADDING)
    return map_x, map_y


def draw_player_on_minimap(minimap, map_x, map_y, track_id):
    cv2.circle(minimap, (map_x, map_y), 5, (0, 0, 255), -1)
    cv2.putText(
        minimap,
        str(track_id),
        (map_x + 6, map_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )


def draw_player_trail(minimap, points):
    if len(points) < 2:
        return

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(minimap, [pts], False, (0, 255, 255), 2)