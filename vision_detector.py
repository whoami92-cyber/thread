import cv2
import os
from ultralytics import YOLO
def maine_threat_vision():
    model_path = "model/best.pt"
    if os.path.exists(model_path): 
        model=YOLO(model_path)
    else:
        print("ladataan YOLO-malli...")
        model=YOLO("yolov8n.pt")
    video_source="video.mp4"
    results = model.predict(source=video_source, stream=True, conf=0.35)
    for r in results:
        annotated_farme = r.plot()
        cv2.imshow("YOLO Detection", annotated_farme)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":
    maine_threat_vision()
    