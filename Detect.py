import torch
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

cap = cv2.VideoCapture(0)

save_dir = 'detected_frames'
os.makedirs(save_dir, exist_ok=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results.render()[0]

    cv2.imshow("Model detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save the current frame
        img_path = os.path.join(save_dir, f"saved_frame_{frame_count:04d}.jpg")
        cv2.imwrite(img_path, annotated_frame)
        print(f"Saved: {img_path}")
        frame_count += 1

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
