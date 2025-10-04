import os
import cv2
from flask import Flask, render_template, Response 
from ultralytics import YOLO 
# --- Import an optimized camera module if using the Pi Camera Module ---
# You would need to install it: pip install picamera2
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Picamera2 not found. Falling back to cv2.VideoCapture(0).")


app = Flask(__name__) 

# --- CONFIGURATION ADJUSTMENTS ---

# 1. Use a smaller, faster model (e.g., YOLOv8n or a quantized model)
# You might need to download 'yolov8n.pt' or an even smaller model 
# that has been optimized/quantized for the Pi's CPU/NPU.
MODEL_PATH = "yolov8n.pt" # Consider yolov8n for better speed on CPU
print(f"Loading model: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH) 
except Exception as e:
    print(f"Error loading model {MODEL_PATH}: {e}")
    # Handle the error or exit gracefully
    model = None


# (COCO names) 
traffic_classes = ['car', 'bus', 'truck', 'motorcycle', 'traffic light', 'stop sign'] 

# --- CAMERA SETUP ADJUSTMENT ---
# Default resolution for a balance of speed and quality on Pi
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

if PI_CAMERA_AVAILABLE:
    # Use Picamera2 for better performance with the CSI camera
    # Initialize the camera
    picam2 = Picamera2()
    # Configure for video (low-resolution for speed)
    picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.start()
    print("Using Picamera2 for camera input.")
else:
    # Fallback to OpenCV VideoCapture for USB webcam
    cap = cv2.VideoCapture(0)
    # Set resolution (may not be respected by all cameras/drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Error: Could not open USB webcam/VideoCapture(0).")
        # Ensure 'cap' is defined even if it fails to open
        cap = None

# -----------------------------------

def get_frame():
    """Captures a single frame from the camera source."""
    if PI_CAMERA_AVAILABLE:
        # Capture from Picamera2
        # Use 'np.ascontiguousarray' to ensure compatibility with OpenCV functions
        frame = picam2.capture_array()
        # Convert the frame from BGR/RGB (depending on config) to the standard BGR for cv2
        # Picamera2 default is RGB, convert to BGR for cv2 processing/display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        return True, frame
    elif cap and cap.isOpened():
        # Capture from OpenCV VideoCapture (USB webcam)
        return cap.read()
    else:
        return False, None


def gen_frames(): 
    if not model:
        print("Cannot run video feed: Model failed to load.")
        return

    while True: 
        success, frame = get_frame() 
        if not success or frame is None: 
            print("Failed to read frame.")
            break 

        # The inference step is the bottleneck on the Pi
        # Set a low confidence threshold for faster filtering, or adjust other params
        # The 'stream=True' argument is more for batch processing, but 'verbose=False' is good.
        results = model(frame, verbose=False, conf=0.5) 
        
        # NOTE: YOLOv8 results structure is slightly different from YOLOv5 in the original code,
        # but the way you extract labels is generally compatible for basic use.
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls] 

        # --- DRAWING & ANNOTATION ---

        is_traffic = any(label in traffic_classes for label in labels) 
        label_text = "Traffic Detected" if is_traffic else "No Traffic" 
        color = (0, 255, 0) if is_traffic else (0, 0, 255) 
        # Use a smaller font size for 640x480 resolution
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 

        # draw bounding boxes 
        # results[0].boxes.xyxy and results[0].boxes.cls are correct for YOLOv8
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls): 
            x1, y1, x2, y2 = map(int, box) 
            cls_name = model.names[int(cls_id)] 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) 
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) 

        # encode frame (JPEG encoding is faster and smaller than PNG) 
        # Use a quality setting if needed for lower bandwidth: cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _, buffer = cv2.imencode('.jpg', frame) 
        frame_bytes = buffer.tobytes() 
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 

#send the frame 2 flask 
@app.route('/') 
def index(): 
    # NOTE: You'll need a simple `index.html` file in a `templates` folder 
    # to display the video feed (e.g., using an `<img>` tag pointing to `/video_feed`).
    return render_template('index.html') 

@app.route('/video_feed') 
def video_feed(): 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 

# --- HOST CONFIGURATION ADJUSTMENT ---
# Bind to '0.0.0.0' so it is accessible from other devices on the local network.
if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# --- CLEANUP (Optional, but good practice) ---
if cap and cap.isOpened():
    cap.release()
if PI_CAMERA_AVAILABLE:
    picam2.stop()
