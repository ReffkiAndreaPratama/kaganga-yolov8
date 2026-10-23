import argparse
import cv2
from ultralytics import YOLO
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sort_boxes_left_to_right(boxes):
    """
    Sort bounding boxes from left to right based on their x1 coordinate.
    This is essential for combining characters into words.
    """
    if len(boxes) == 0:
        return boxes
    # Sort by x1 (index 0)
    sorted_indices = np.argsort(boxes[:, 0])
    return boxes[sorted_indices]

def main(weights_path, conf_thres, camera_id):
    logger.info(f"Loading model from {weights_path}...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Get class names dictionary from model
    class_names = model.names
    
    logger.info(f"Starting webcam (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Could not open webcam {camera_id}")
        return

    # Request high resolution from camera (adjust as needed for fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("Webcam started successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame.")
            break
            
        # Run YOLO inference
        # verbose=False reduces console spam
        results = model.predict(frame, conf=conf_thres, verbose=False)
        
        for result in results:
            # Extract box data
            boxes = result.boxes.data.cpu().numpy() # shape: [N, 6] -> [x1, y1, x2, y2, conf, cls]
            
            if len(boxes) > 0:
                # Sort boxes left-to-right to read characters in order
                sorted_boxes = sort_boxes_left_to_right(boxes)
                
                detected_word = ""
                
                # Draw boxes and build the final word
                for box in sorted_boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    class_name = class_names[int(cls_id)]
                    
                    # Append character to word string
                    detected_word += class_name + " "
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name} {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Background for text
                    cv2.rectangle(frame, 
                                  (int(x1), int(y1) - text_height - 10), 
                                  (int(x1) + text_width, int(y1)), 
                                  (0, 255, 0), 
                                  -1)
                    # Text itself
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Display the combined detected word at the top of the frame
                cv2.putText(frame, f"Text: {detected_word.strip()}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow("Realtime Character OCR", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Realtime OCR stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime YOLOv8 Inference for OCR")
    parser.add_argument("--weights", type=str, default="runs/detect/train_optimized/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (increase to reduce false positives)")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (0 for default webcam)")
    args = parser.parse_args()
    
    main(args.weights, args.conf, args.camera)
