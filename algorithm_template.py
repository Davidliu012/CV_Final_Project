import os
import json
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
    model = YOLO('yolov8n.pt')
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

def detect_door(img, model, confidence_threshold=0.78):
    """Detects door in an image using YOLOv8."""
    results = model(img)
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    print(results[0])
    door_index = None
    for idx, name in model.names.items():
        if name == 'door':
            door_index = idx
            break

    # if door_index is None:
    #     print("Door class not found in the model.")
    #     return None
    
    # Iterate through detected objects
    for i, box in enumerate(boxes):
        # if class_ids[i] == door_index and confidences[i] > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"door: {confidences[i]:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite('output/yolo.jpg', img)
            return (x1, y1, x2-x1, y2-y1)  # Return the first detected door box (x, y, w, h)
    
    return None
# def detect_door(img, net, output_layers, classes, confidence_threshold=0.5, nms_threshold=0.4):
#     """Detects door in an image using YOLO."""
#     height, width = img.shape[:2]
    
#     # Create a blob from the image
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
    
#     # Initialize lists for detected bounding boxes, confidences, and class IDs
#     boxes = []
#     confidences = []
#     class_ids = []
    
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > confidence_threshold and classes[class_id] == 'door':
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     # Apply non-maxima suppression to remove multiple boxes for the same object
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
#     if len(indices) > 0:
#         box = boxes[indices[0][0]]
#         return box  # Return the first detected door box (x, y, w, h)
#     else:
#         return None
    
def door_opening_detected(img1, img2, motion_threshold=2.0, feature_params=None, lk_params=None):
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
  
    
    # Detect good features to track
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    if p0 is None:
        return False

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
    
    # Detect if the motion exceeds the threshold
    if mean_motion > motion_threshold:
        return True
    
    return False

def guess_door_opening(video_filename):
    """Simulate guessing the frame for door opening."""
    # Hypothetical function: replace with actual logic.
    cap = cv2.VideoCapture(video_filename)
    fps = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    print(fps)
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

    cap.release()
    return target_frame


def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    videos_info = []

    for video_file in video_files:
        if video_file == "09.mp4":
            continue
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
    directory = "./Test Videos"  # Specify the directory to scan
    if(directory == "./Sample Videos"):
        output_filename = "output/Sample/algorithm_output.json"  # Output JSON file name
    elif(directory == "./Test Videos"):
        output_filename = "output/Test/algorithm_output.json"  # Output JSON file name
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")


if __name__ == "__main__":
    main()
