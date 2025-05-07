from ultralytics import YOLO
import cv2
from CoreLocation import CLLocationManager, kCLAuthorizationStatusAuthorizedAlways
import argparse
import time


# Parse command-line arguments
parser = argparse.ArgumentParser(description="YOLO Tracking with Location Logging")
parser.add_argument('--name', type=str, required=True, help='Name to include in logging')
args = parser.parse_args()
name = args.name

class LocationManager:
    def __init__(self):
        self.manager = CLLocationManager.alloc().init()
        self.manager.requestAlwaysAuthorization()
        self.manager.startUpdatingLocation()
    
    def get_location(self):
        loc = self.manager.location()
        if loc:
            return f"Latitude: {loc.coordinate().latitude}, Longitude: {loc.coordinate().longitude}"
        else:
            return "Location not available"

# Load the YOLO model
model = YOLO('weights.pt')

# Open webcam (live video)
webcamera = cv2.VideoCapture(0)
webcamera.set(cv2.CAP_PROP_FPS, 5)

loc_manager = LocationManager()

# Check if the camera opened successfully
if not webcamera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Video properties
width = int(webcamera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 5

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(f'realtime-res/{name}.mp4', fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

ids = []
frame_id = 0

with open(f'realtime-res/{name}.txt', 'w') as f:
    while True:
        frame_id += 1
        success, frame = webcamera.read()
        if not success:
            print("Frame read failed. Attempting to reinitialize camera.")
            webcamera.release()
            time.sleep(1)
            webcamera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            continue

        results = model.track(source=frame, persist=True)
        coordinates = loc_manager.get_location()

        for result in results:
            for obj in result.boxes:
                obj_id = obj.id
                bbox = obj.xywh.tolist()[0]
                confidence = obj.conf.item()
                label = obj.cls.item()
                if obj_id is not None:
                    obj_id = obj_id.item()
                    ids.append(obj_id)
                f.write(f"frame: {frame_id}, ID: {obj_id}, BBox: {bbox}, Confidence: {confidence}, Label: {label}, Coordinates: {coordinates}\n")
             
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Tracking", annotated_frame)
        out.write(annotated_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:  
            break


# Cleanup
webcamera.release()
cv2.destroyAllWindows()
out.release()
