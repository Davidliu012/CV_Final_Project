import cv2

# Open the video file
cap = cv2.VideoCapture('Sample Videos/01.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the frame count
frame_count = 0

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Frame number: {frame_count}")
    frame_count += 1

# Release the video capture object and print the frame count
cap.release()
print(f"Total number of frames: {frame_count}")