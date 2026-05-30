import cv2
from pathlib import Path


def extract_frames(video_path:str, output_folder:str, every_n_frames:int=30):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = video.read()

        if not ret:
            print("No more frames or failed to read video.")
            break
        if frame_id % every_n_frames == 0:
            output_path = output_folder / video_path.stem / f"frame_{saved_id:03d}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, frame)
            saved_id += 1
        
        frame_id += 1
    
    video.release()

def has_text_overlay(frame, y1=20, y2=95, x1=35, x2=970, threshold=0.03):
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_mask = gray > 200

    bright_ratio = bright_mask.mean()

    return bright_ratio > threshold

def blur_text_region_if_needed(frame, y1=20, y2=95, x1=35, x2=970, height=720, width=1280):
    clean = frame.copy()
    bottom_y1 = height-y2 - 25
    bottom_y2 = height-y1 - 25
    buttom_x1 = width-x2
    bottom_x2 = width-x1

    if has_text_overlay(frame, y1, y2, x1, x2):
        roi = clean[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (99, 99), 0)
        clean[y1:y2, x1:x2] = blurred

    elif has_text_overlay(frame, bottom_y1, bottom_y2, buttom_x1, bottom_x2):
        roi = clean[bottom_y1:bottom_y2, buttom_x1:bottom_x2]
        blurred = cv2.GaussianBlur(roi, (99, 99), 0)
        clean[bottom_y1:bottom_y2, buttom_x1:bottom_x2] = blurred

    return clean