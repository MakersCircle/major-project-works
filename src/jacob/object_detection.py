import cv2
from ultralytics import YOLO
import os

def process_video(input_video_path, output_dir="./sample"):
    """
    Process a video using YOLO object detection and save it to the specified output directory.

    Args:
        input_video_path (str): Path to the input video.
        output_dir (str): Directory where the processed video will be saved.

    Returns:
        str: Path to the processed video if successful, None otherwise.
    """
    model = YOLO("yolov8m.pt")

    # Extract filename without extension
    video_name = os.path.basename(input_video_path)
    video_name_without_ext, _ = os.path.splitext(video_name)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output video path
    output_video_path = os.path.join(output_dir, f"{video_name_without_ext}_object.mp4")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(frame)
    except Exception as e:
        print(f"Error during video processing: {e}")
        return None
    finally:
        cap.release()
        out.release()

    print(f"Processed video saved at: {output_video_path}")
    return output_video_path
