import cv2
import time
from pathlib import Path

from mlbox.settings import ROOT_DIR
from datetime import datetime

CURRENT_DIR = Path(__file__).parent

# Define the RTSP stream URL
rtsp_url = "rtsp://admin:Hik_vision@10.11.12.66:554/ISAPI/Streaming/Channels/1"
#rtsp_url = "rtsp://admin:Admin12345@10.11.198.159:554/Streaming/Channels/1"

interval = 1
camera_name = "0713"

# Define the start and end times
START_TIME = datetime.strptime("02:30", "%H:%M").time()
END_TIME = datetime.strptime("19:00", "%H:%M").time()


last_saved_time = time.time()
output_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"
output_folder.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(rtsp_url)
if cap.isOpened():
    print("Start capturing frames!")
cap.release()

while True:

    # Check if the interval has passed
    if START_TIME <= datetime.now().time() <= END_TIME:
        
        cap = cv2.VideoCapture(rtsp_url)

        # Check if the video capture object was successfully created
        if not cap.isOpened():
            print("Error: Could not open video stream")
            break

        ret, frame = cap.read()
        cap.release()
        
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        
        filename = f"{camera_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
        cv2.imwrite(output_folder / filename, frame)
        
        #time.sleep(interval)


cv2.destroyAllWindows()
