import requests
import cv2
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import sys

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mlbox.settings import ROOT_DIR

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "camera_capture.log")
    ]
)

# Camera settings
camera_ip = "10.11.17.25"
camera_username = "admin"
camera_password = "Hik_vision"
camera_name = "0713"
snapshot_url = f"http://{camera_ip}/ISAPI/Streaming/channels/1/picture"

# Capture settings
capture_interval = 1.5  # seconds between captures

# Time window settings - using current date for context
current_date = datetime.now().date()
START_TIME = datetime.strptime("02:30", "%H:%M").time()
END_TIME = datetime.strptime("17:30", "%H:%M").time()

# Setup output folder with date-based structure
output_base = ROOT_DIR / "tmp" / Path(__file__).parent.name / "output"
today_folder = output_base / current_date.strftime("%Y-%m-%d")
today_folder.mkdir(parents=True, exist_ok=True)

print(f"Starting camera capture on {current_date}. Images will be saved to {today_folder}")
print(f"Taking snapshots every {capture_interval} seconds")
print(f"Capture window: {START_TIME} to {END_TIME}")
print(f"Using URL: {snapshot_url}")


try:
    capture_count = 0
    error_count = 0
    session = requests.Session()  # Use a session for better performance
    
    while True:
        # Check for new day and create folder if needed
        current_date = datetime.now().date()
        current_time = datetime.now().time()
        today_folder = output_base / current_date.strftime("%Y-%m-%d")
        today_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if we're in the capture time window
        if START_TIME <= current_time <= END_TIME:
            try:
                # Get current time before the request
                start_time = time.time()
                frame_capture_time = datetime.now()
                
                # Using digest authentication explicitly
                response = session.get(
                    snapshot_url, 
                    auth=requests.auth.HTTPDigestAuth(camera_username, camera_password), 
                    timeout=5
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    # Reset error counter on success
                    error_count = 0
                    
                    # Convert response content to image
                    image_data = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Create filename with timestamp
                        filename = f"{camera_name}_{frame_capture_time.strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
                        file_path = today_folder / filename
                        
                        # Save the image
                        cv2.imwrite(str(file_path), frame)
                        
                        capture_count += 1
                        
                    else:
                        print("Error: Could not decode image from response")
                else:
                    error_count += 1
                    print(f"Error: HTTP request failed with status code {response.status_code}")
                    print(f"Response: {response.text[:100]}...")  # Print first 100 chars of response
                    
                    # If we get several errors in a row, wait longer
                    if error_count > 3:
                        wait_time = min(30, error_count * 5)  # Gradually increase wait time
                        print(f"Multiple errors occurred. Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
                    
                # Calculate time to sleep to maintain the capture interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, capture_interval - elapsed)
                time.sleep(sleep_time)
                    
            except requests.exceptions.RequestException as e:
                error_count += 1
                print(f"Error: HTTP request failed: {e}")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(5)  # Wait before retrying
        else:
            # Outside capture window
            current_time = datetime.now().time()
            print(f"Outside capture window. Current time: {current_time}. Next check in 60 seconds.")
            time.sleep(60)  # Check every minute when outside the capture window
            
except KeyboardInterrupt:
    print(f"\nCapture stopped by user after saving {capture_count} frames")
except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Camera capture script terminated")
