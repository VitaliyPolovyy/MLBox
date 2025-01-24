import cv2

# Define the RTSP stream URL
rtsp_url = "rtsp://admin:Hik_vision@10.11.12.66:554/ISAPI/Streaming/Channels/1"

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# Check if the video capture object was successfully created
if not cap.isOpened():
    print("Error: Could not open video stream")
else:
    print("Trying to open the stream")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
