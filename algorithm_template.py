import os
import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def parseString(str):
    # Split the path to get the folder and the filename
    path_parts = str.split('/')

    # Extract the folder name and take the first word
    folder = path_parts[1].split(' ')[0]

    # Extract the filename and remove the extension
    filename_with_extension = path_parts[2]
    number = filename_with_extension.split('.')[0]
    # print(str, folder, number)
    return folder, number

def get_door_box(img):
    # Load YOLO
    # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # layer_names = net.getLayerNames()
    # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # classes = []
    # with open("coco.names", "r") as f:
    #     classes = [line.strip() for line in f.readlines()]
    # door_box = detect_door(img, net, output_layers, classes)
    model = YOLO('yolov8_20epoch.pt')
    door_box = detect_door(img, model)
    if door_box is not None:
        # Save the detected door image
        x, y, w, h = door_box
        door_img = img[y:y+h, x:x+w]
        cv2.imwrite("detected_door.jpg", door_img)
        # print("Door opening detected")
    else:
        print("No door detected in the first frame.")
    
    return door_box

def detect_door(img, model, confidence_threshold=0.6):
    """Detects door in an image using YOLOv8."""
    results = model(img)
    boxes = results[0].boxes.xyxy
    # print("boxes: ", boxes)
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    # print("result: ", results[0])
    door_index = None
    # print(model.names.items())
    for idx, name in model.names.items():
        if name == 'bus-door':
            door_index = idx
            break

    if door_index is None:
        print("Door class not found in the model.")
        # return None
    
    # Iterate through detected objects
    for i, box in enumerate(boxes):
        if class_ids[i] == door_index and confidences[i] >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            print(f"Door detected at ({x1}, {y1}, {x2}, {y2}) with confidence {confidences[i]:.2f}")
            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"door: {confidences[i]:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite('output/yolo.jpg', img)
            return (x1, y1, x2-x1, y2-y1)  # Return the first detected door box (x, y, w, h)
    
    # return None
    h, w, c = img.shape
    cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 2)
    
    # Place the text at the top-left corner within the image bounds
    cv2.putText(img, "No door", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite('output/yolo.jpg', img)
    return (0, 0, w, h)
    
def door_opening_detected(img1, img2, motion_threshold=2.5, feature_params=None, lk_params=None):
    """Detects door opening based on optical flow between two frames."""
    if feature_params is None:
        feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=10)
    
    if lk_params is None:
        lk_params = dict(winSize=(150, 150), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    # Get door bounding boxes
    door_box = get_door_box(img1)
    x, y, w, h = door_box
    roi1 = img1[y:y+h, x:x+w]
    roi2 = img2[y:y+h, x:x+w]
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    
    # gray1 = roi1
    # gray2 = roi2
    
    # # Detect good features to track
    # p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    # if p0 is None:
    #     return False
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    
    if len(kp1) == 0:
        return False
    
    # Convert keypoints to points
    p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    
    if p1 is None or st is None:
        return False
    
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Calculate movement
    motion = np.sqrt((good_new[:, 0] - good_old[:, 0]) ** 2 + (good_new[:, 1] - good_old[:, 1]) ** 2)
    mean_motion = np.mean(motion)
    
    # Visualize the keypoints and motion vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)  # Convert coordinates to integers
            roi1 = cv2.circle(roi1, (a, b), 5, (0, 255, 0), -1)
            roi1 = cv2.line(roi1, (a, b), (c, d), (255, 0, 0), 2)
        
    # Place the ROI back into the original image
    img1[y:y+h, x:x+w] = roi1
    cv2.imwrite('output/keypoints.jpg', img1)
    
    # Detect if the motion exceeds the threshold
    if mean_motion > motion_threshold:
        return True
    
    return False

def guess_door_opening(video_filename):
    """Simulate guessing the frame for door opening."""
    # Hypothetical function: replace with actual logic.
    cap = cv2.VideoCapture(video_filename)
    fps = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    # print(fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = -1
    ret, previous_frame = cap.read()
    for frame_index in range(1, frame_count):
        ret, current_frame = cap.read()
        if not ret:
            break
        
        if door_opening_detected(previous_frame, current_frame):
            print("Door opening detected at frame:", frame_index)
            target_frame = frame_index
            folder, number = parseString(video_filename)
            cv2.imwrite(f'output/{folder}/door_opening_frame_{number}.jpg', current_frame)
            break
        
        previous_frame = current_frame
        print(f"Opening Frame {frame_index}/{frame_count} processed.")
        print(f"Opening Frame detected at frame:", target_frame)

    cap.release()
    return target_frame


def guess_door_closing(video_filename):
    """Simulate guessing the frame for door closing."""
    # Hypothetical function: replace with actual logic.
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = -1
    
    # Start from the end of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, previous_frame = cap.read()
    
    for frame_index in range(frame_count - 2, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, current_frame = cap.read()
        if not ret:
            break
        
        if door_opening_detected(previous_frame, current_frame):
            print("Door closing detected at frame:", frame_index)
            target_frame = frame_index
            folder, number = parseString(video_filename)
            cv2.imwrite(f'output/{folder}/door_closing_frame_{number}.jpg', current_frame)
            break
        
        previous_frame = current_frame
        
        previous_frame = current_frame
        print(f"Closing Frame {frame_index}/{frame_count} processed.")
        print(f"Closing Frame detected at frame:", target_frame)

    cap.release()
    return target_frame


def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    videos_info = []

    for video_file in video_files:
        if video_file == "01.mp4":
            
            videos_info.append(
            {
                "video_filename": video_file,
                "annotations": [
                    {
                        "object": "Door",
                        "states": [
                            {
                                "state_id": 1,
                                "description": "Opening",
                                # "start_frame": 100,
                                # "half_open_frame": 130,
                                "guessed_frame": guess_door_opening(
                                    directory + "/" + video_file
                                ),  # Guessing frame using function.
                            },
                            {
                                "state_id": 2,
                                "description": "Closing",
                                # "start_frame": 178,
                                # "end_frame": 220,
                                "guessed_frame": guess_door_closing(
                                    directory + "/" + video_file
                                ),  # Guessing frame using function.
                            },
                        ],
                    }
                ],
            }
        )

    return videos_info


def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, "w") as file:
        json.dump({"videos": videos_info}, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Process video directories.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', action='store_true', help='Use the Sample Videos directory')
    group.add_argument('--test', action='store_true', help='Use the Test Videos directory')

    args = parser.parse_args()

    if args.sample:
        directory = "./Sample Videos"
        output_filename = "output/Sample/algorithm_output.json"
    elif args.test:
        directory = "./Test Videos"
        output_filename = "output/Test/algorithm_output.json"
        
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")


if __name__ == "__main__":
    main()
