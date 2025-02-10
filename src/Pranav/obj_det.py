import cv2
import torch
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolov8m.pt")

    def detect_objects(self, frame, n=5):
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes.data.tolist():
            if len(box) < 6:
                continue  # Skip invalid detections
            x1, y1, x2, y2, conf, cls = box[:6]
            detections.append([int(x1), int(y1), int(x2), int(y2), conf, int(cls)])

        detections = sorted(detections, key=lambda x: x[4], reverse=True)[:n]
        return detections

    def process_video(self, video_path, n=5):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frames_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_objects(frame, n)
            frames_data.append(detections)

            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                label = f"{self.model.names[cls]} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return frames_data


if __name__ == "__main__":
    video_path = "C:/Users/Pranav/Documents/Programming/major-project-works/sample/000001.mp4"
    detector = ObjectDetection()
    frames_data = detector.process_video(video_path, n=5)
