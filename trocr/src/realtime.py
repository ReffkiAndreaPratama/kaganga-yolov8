import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import time

def main(model_path, camera_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TrOCR model on {device}...")
    
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Request high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("========================================")
    print("Press 'Space' to trigger OCR on the current frame.")
    print("Press 'q' to quit.")
    print("========================================")

    current_prediction = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        display_frame = frame.copy()
        
        # Draw a Region of Interest (ROI) box in the center for the user to align the character
        h, w, _ = display_frame.shape
        roi_width, roi_height = 400, 200
        x1, y1 = (w - roi_width) // 2, (h - roi_height) // 2
        x2, y2 = x1 + roi_width, y1 + roi_height
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Place character/word in box & Press SPACE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw prediction result
        if current_prediction:
            cv2.putText(display_frame, f"Prediction: {current_prediction}", (10, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("TrOCR Realtime", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '): # Spacebar triggered
            # Crop ROI
            roi = frame[y1:y2, x1:x2]
            
            # Convert to PIL Image for TrOCR
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(roi_rgb)
            
            # Inference with automatic mixed precision
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
            
            start_time = time.time()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                    generated_ids = model.generate(pixel_values)
            end_time = time.time()
            
            current_prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Detected: '{current_prediction}' in {end_time-start_time:.2f}s")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realtime TrOCR")
    parser.add_argument("--model", type=str, default="../models/checkpoints/best_model", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    args = parser.parse_args()
    
    main(args.model, args.camera)
