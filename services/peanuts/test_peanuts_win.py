"""
Test script for Peanuts service - Windows version
Run this from Windows CMD or PowerShell: python test_peanuts_win.py
"""
import requests

# Configuration
SERVER_URL = "http://10.11.122.100:8001/peanuts/process_image"
IMAGE_PATH = r"C:\Users\vitaliy.polovoy\Downloads\2\D21010101000026_12835_C9601123601.jpg"

# Request payload
request_data = {
    "service_code": "1",
    "alias": "DMS",
    "key": "   9127673     1",
    "response_method": "HTTP_POST_REQUEST",
    "response_endpoint": "https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false"
}

# Send request
print(f"Sending request to: {SERVER_URL}")
print(f"Image: {IMAGE_PATH}")
print("-" * 60)

try:
    with open(IMAGE_PATH, 'rb') as image_file:
        r = requests.post(
            SERVER_URL,
            files={'image': image_file},
            data={'json': str(request_data).replace("'", '"')}
        )
    
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.text}")
    
except FileNotFoundError:
    print(f"Error: Image file not found at {IMAGE_PATH}")
    print("Please update IMAGE_PATH variable with the correct path to your image.")
except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to server at {SERVER_URL}")
    print("Please check if the server is running and accessible.")
except Exception as e:
    print(f"Error: {e}")

