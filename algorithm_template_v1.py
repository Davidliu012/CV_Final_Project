import os
import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

def parseString(str):
    # Split the path to get the folder and the filename
    path_parts = str.split("/")

    # Extract the folder name and take the first word
    folder = path_parts[1].split(" ")[0]

    # Extract the filename and remove the extension
    filename_with_extension = path_parts[2]
    number = filename_with_extension.split(".")[0]
    # print(str, folder, number)
    return folder, number


def get_door_mask(img):
    # Load YOLO segmentation model
    model = YOLO("yolov8_segmentation.pt")
    detected_objects = detect_door(img, model)
    
    door_mask = []

    for obj, mask in detected_objects:
        if obj == 'door':
            door_mask.append(mask)
            
    black_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # print(door_mask)
    mask_area = 0
    if door_mask != []:
        for mask in door_mask:
            cv2.fillPoly(black_img, mask, (255, 255, 255))
            mask_area += np.sum(mask)
        
    else:
        # black_img = np.full_like(black_img, 255)
        mask_area = black_img.shape[0] * black_img.shape[1]
        # print(black_img)
        print("No door detected in the first frame.")
        
    cv2.imwrite("output/detected_door.jpg", black_img)

    return black_img, mask_area


def detect_door(img, model, confidence_threshold=0.6):
    """Detects door in an image using YOLOv8."""
    results = model(img, save_txt=True)
    detected_objects = []
    # print("results: ", results)
    
    # Find the indices for door and people classes
    for idx, name in model.names.items():
        if name == "bus-door":
            door_index = idx
            
    if door_index is None:
        print("Door class not found in the model.")

    # colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # print(results)
    global last_door_mask
    
    for result in results:
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                
                if confidence >= confidence_threshold:
                    if class_id == door_index:
                        print(f"Door detected with confidence {confidence:.2f}")
                        detected_objects.append(('door', points))
                        # last_door_mask = points
                        cv2.fillPoly(img, points, (255, 255, 255))
        
    
    cv2.imwrite("output/yolo_segmentation.jpg", img)
    return detected_objects

def detect_door_box(img, model, confidence_threshold=0.6):
    """Detects door in an image using YOLOv8."""
    results = model(img)
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    door_index = None
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

def get_door_box(img):
    model = YOLO('yolov8_box.pt')
    door_box = detect_door_box(img, model)
    if door_box is not None:
        # Save the detected door image
        x, y, w, h = door_box
        door_img = img[y:y+h, x:x+w]
        cv2.imwrite("detected_door.jpg", door_img)
        # print("Door opening detected")
    else:
        print("No door detected in the first frame.")
    
    return door_box

def door_opening_detected(
    img1, img2, motion_threshold=3.5, feature_params=None, lk_params=None
):
    """Detects door opening based on optical flow between two frames."""
    if feature_params is None:
        feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=10
        )

    if lk_params is None:
        lk_params = dict(
            winSize=(150, 150),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    # Get door bounding boxes
    door_box = get_door_box(img1)
    x, y, w, h = door_box
    roi1 = img1[y : y + h, x : x + w]
    roi2 = img2[y : y + h, x : x + w]
    
    # roi1 = img1
    # roi2 = img2
    # Convert images to grayscale
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    
    # Get door mask
    door_mask, mask_area = get_door_mask(roi1)
    
    motion_threshold = motion_threshold * (mask_area / (img1.shape[0] * img1.shape[1]))  # Scale the threshold by the door mask area
    print("motion_threshold", motion_threshold, mask_area)
    # Apply door mask to the images
    roi1 = cv2.bitwise_and(gray1, gray1, mask=door_mask)
    roi2 = cv2.bitwise_and(gray2, gray2, mask=door_mask)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)

    if len(kp1) == 0:
        return False, 0
    
    # Filter out keypoints that fall outside the door mask
    if door_mask is None:
        p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)

    else:
        valid_keypoints = [kp for kp in kp1 if door_mask[int(kp.pt[1]), int(kp.pt[0])] == 255]
        if not valid_keypoints:
            print("Invalid keypoints")
            p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)
            # return False
        else:
            # Convert valid keypoints to points
            p0 = np.array([kp.pt for kp in valid_keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # # Convert keypoints to points
    # p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow
    p1, status, error = cv2.calcOpticalFlowPyrLK(roi1, roi2, p0, None, **lk_params)

    if p1 is None or status is None:
        return False, 0

    # Select good points
    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # Calculate movement
    motion = np.sqrt(
        (good_new[:, 0] - good_old[:, 0]) ** 2 + (good_new[:, 1] - good_old[:, 1]) ** 2
    )
    mean_motion = np.mean(motion)

    # # Visualize the keypoints and motion vectors
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     a, b, c, d = int(a), int(b), int(c), int(d)  # Convert coordinates to integers
    #     img1 = cv2.circle(img1, (a, b), 5, (0, 255, 0), -1)
    #     img1 = cv2.line(img1, (a, b), (c, d), (255, 0, 0), 2)

    # # Place the ROI back into the original image
    # img1[y : y + h, x : x + w] = roi1
    
    # Visualize the keypoints and motion vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a + x), int(b + y), int(c + x), int(d + y)
        img1 = cv2.circle(img1, (a, b), 5, (0, 255, 0), -1)
        img1 = cv2.line(img1, (a, b), (c, d), (255, 0, 0), 2)

    cv2.imwrite("output/keypoints.jpg", img1)

    return mean_motion > motion_threshold, mean_motion


def guess_door_opening_closing(video_filename):
    """Simulate guessing the frame for door opening."""
    # Hypothetical function: replace with actual logic.
    cap = cv2.VideoCapture(video_filename)
    fps = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = []
    ret, previous_frame = cap.read()
    previous_frame = cv2.resize(
        previous_frame, (previous_frame.shape[1] // 2, previous_frame.shape[0] // 2)
    )
    state_list = []
    mean_motions = []
    frame_indices = []
    folder, number = parseString(video_filename)

    for frame_index in range(1, frame_count):
        ret, current_frame = cap.read()
        current_frame = cv2.resize(
            current_frame, (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
        )
        if not ret:
            break
        
        mean_motion_detected, mean_motion = door_opening_detected(previous_frame, current_frame)
        if(math.isnan(mean_motion)):
            mean_motion = 0
        print("mean motion detected", mean_motion)
        mean_motions.append(mean_motion)
        frame_indices.append(frame_index)

        previous_frame = current_frame
        print(f"Opening Frame {frame_index}/{frame_count} processed.")

    cap.release()
    
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, mean_motions, marker="o", linestyle="-", color="b")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Motion")
    plt.title("Mean Motion Over Frames")
    plt.grid(True)
    
    print("mean motions", mean_motions)
    target_frames = get_guessed_frame_list(mean_motions)
    plt.savefig(f"output/mean_motion_over_frames_{number}.png")

    # # Plotting the state transitions
    # plt.figure(figsize=(12, 6))
    # plt.plot(
    #     range(1, len(state_list) + 1), state_list, marker="o", linestyle="-", color="b"
    # )
    # plt.xlabel("Frame Index")
    # plt.ylabel("State")
    # plt.title("State Transitions Over Frames")
    # plt.grid(True)

    # # Save the plot as an image
    # plt.savefig(f"output/{folder}/state_transitions_{number}.png")
    # plt.close()
    
    return target_frames

def filter_small_motion(mean_motions):
    # Filter out small motion values
    filtered_motions = []
    threshold = 0.5
    for motion in mean_motions:
        if motion < threshold:
            filtered_motions.append(0)
        else:
            filtered_motions.append(motion)
        
    return filtered_motions

def nan_filtered_motions(mean_motions):
    filtered_motions = []
    window_size = 8
    for i in range(len(mean_motions)):
        if mean_motions[i] == -1:
            window = mean_motions[
                max(0, i - window_size // 2) : min(
                    len(mean_motions), i + window_size // 2
                )
            ]
            filtered_motions.append(max(0, np.max(window)))

    return filtered_motions

def round_filtered_motions(mean_motions):
    filtered_motions = []
    for i in range(len(mean_motions)):
        filtered_motions.append(round(mean_motions[i] / 0.1) * 0.1)
    return filtered_motions

def median_filtered_motions(mean_motions):
    # Apply median filter to the mean motion values
    filtered_motions = []
    window_size = 8
    for i in range(len(mean_motions)):
        window = mean_motions[
            max(0, i - window_size // 2) : min(
                len(mean_motions), i + window_size // 2
            )
        ]
        filtered_motions.append(np.median(window))

    return filtered_motions

def get_guessed_frame_list(mean_motions):
    # mean_motions = nan_filtered_motions(mean_motions)
    # mean_motions = median_filtered_motions(mean_motions)
    mean_motions = filter_small_motion(mean_motions)

    # Find peaks in the mean_motion array
    window = np.zeros((len(mean_motions), 2), dtype=np.float32)
    window_size = 30

    for i in range(len(mean_motions)):
        # Calculate the mean motion in the window and store it
        window[i, 0] = np.mean(mean_motions[max(0, i - window_size // 2):min(len(mean_motions), i + window_size // 2)])
        window[i, 0] = window[i, 0] + mean_motions[i] * 0.3
        # Store the corresponding frame index
        window[i, 1] = i

    # sort from large to small
    original_window = np.copy(window)
    window = window[window[:, 0].argsort()[::-1]]

    # hyper_params

    forbiddened_range = 70

    # get the k-largest points that keep a distance "forbiddened_range" from each other
    max_list = []
    forbiddened_list = []
    max_motion = window[0][0]
    decay_term = 0.1 * max_motion
    threshold = max_motion - decay_term

    for i in range(len(window)):
        if len(max_list) >= 4:
            break

        if (window[i][0] < threshold):
            if (len(max_list) % 2 == 1):
                threshold = threshold - decay_term
            elif (len(max_list) % 2 == 0):
                break
        frame_index = window[i][1]

        if frame_index not in forbiddened_list:
            selected_frame = 0

            max_list.append(frame_index)

            forbiddened_list.extend(
                range(
                    int(frame_index - forbiddened_range),
                    int(frame_index + forbiddened_range),
                )
            )

    max_list = np.sort(max_list).tolist()
    for j in range(len(max_list)):
        original_frame_index = int(max_list[j])

        # compute range_length
        range_length = window_size
        front_range_length = 0
        back_range_length = 0
        # compute back_range_length
        for k in range(
            original_frame_index + 1,
            min(
                len(window) - 1,
                int(max_list[min(j + 1, len(max_list) - 1)]) + forbiddened_range,
            ),
        ):
            if original_window[k][0] < 0.2 * original_window[original_frame_index][0]:
                back_range_length = k - original_frame_index
                print("back_range_length: ", back_range_length)
                break

        # compute front_range_length
        for k in range(
            original_frame_index - 1,
            max(
                0,
                int(max_list[max(j - 1, 0)]) - forbiddened_range,
            ),
            -1
        ):
            if original_window[k][0] < 0.2 * original_window[original_frame_index][0]:
                front_range_length = original_frame_index - k
                print("front_range_length: ", front_range_length)
                break

        range_length = (front_range_length + back_range_length) / 2

        if (j % 2 == 1):
            selected_frame = min(
                len(window) - 1, int(max_list[j] + (1 / 6) * range_length)
            )
            max_list[j] = selected_frame

        elif (j % 2 == 0):
            selected_frame = max(0, int(max_list[j] - (1 / 6) * range_length))
            max_list[j] = selected_frame

    return max_list

def guess_peak_frames(mean_motions, frame_indices):
    # Find peak ranges in the mean_motion array
    peak_ranges = get_guessed_frame_list(mean_motions)
    
    if len(peak_ranges) < 2:
        return []

    guessed_frames = []
    
    # For the first peak range, select the first 1/3 of the frames
    first_peak_range = peak_ranges[0]
    first_third_index = len(first_peak_range) // 3
    guessed_frames.append(frame_indices[first_peak_range[first_third_index]])
    
    # For the second peak range, select the last 1/3 of the frames
    if len(peak_ranges) > 1:
        second_peak_range = peak_ranges[1]
        last_third_index = len(second_peak_range) * 2 // 3
        guessed_frames.append(frame_indices[second_peak_range[last_third_index]])
    
    return guessed_frames

def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    videos_info = []

    for video_file in video_files:
        if video_file == "11.mp4":
            print("starting video " + video_file)
            states = []
            target_frames = guess_door_opening_closing(directory + "/" + video_file)
            # target_frames = 
            # print(target_frames)
            for n in range(0, len(target_frames), 2):
                states.append(
                    {
                        "state_id": n + 1,
                        "description": "Opening",
                        "guessed_frame": target_frames[
                            n
                        ], 
                    }
                )
                if n + 1 < len(target_frames):
                    states.append(
                        {
                            "state_id": n + 2,
                            "description": "Closing",
                            "guessed_frame": target_frames[
                                n + 1
                            ], 
                        }
                    )

            videos_info.append(
                {
                    "video_filename": video_file,
                    "annotations": [
                        {
                            "object": "Door",
                            "states": states,
                        }
                    ],
                }
            )
    print(videos_info)

    return videos_info


def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, "w") as file:
        json.dump({"videos": videos_info}, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process video directories.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample", action="store_true", help="Use the Sample Videos directory"
    )
    group.add_argument(
        "--test", action="store_true", help="Use the Test Videos directory"
    )

    args = parser.parse_args()

    if args.sample:
        directory = "./Sample Videos"
        output_filename = "output/Sample/algorithm_output.json"
    elif args.test:
        directory = "./Test Videos"
        output_filename = "output/Test/algorithm_output.json"

    mean_motion_01 = [0.04707209, 0.07189061, 0.10222822, 0.21687351, 0.12493976, 0.1267368, 0.22330917, 0.30494764, 0.1250756, 0.0466429, 0.050375327, 0.08269501, 0.12373447, 0.03311016, 0.07798707, 0.05091782, 0.14114088, 0.02900462, 0.03632169, 0.12879406, 0.06584638, 0.20685978, 0.28707758, 0.30821392, 0.030386347, 0.17745811, 0.044102307, 0.048268255, 0.059594244, 0.06365809, 0.084823295, 0.10949211, 0.069561444, 0.20190209, 0.040377382, 0.34085003, 0.2709912, 0.039058045, 0.031801287, 0.102658294, 0.07061571, 0.22021726, 0.077610716, 0.11154079, 0.045947086, 0.018919008, 0.09407315, 0.039900605, 0.040563807, 0.10221874, 0.030582448, 0.11007562, 0.05578302, 0.047262505, 0.03503911, 0.060295533, 0.028533582, 0.08642767, 0.04115346, 0.043488353, 0.030100254, 0.029191146, 0.025534315, 0.05709178, 0.018192101, 0.045974225, 0.024496391, 0.023556257, 0.028918127, 0.030130545, 0.031150574, 0.03694513, 0.032157633, 0.08739606, 0.1082909, 0.15746675, 0.28249094, 0.38836083, 1.5666578, 3.325454, 2.5598083, 3.3863623, 0.37099555, 3.611042, 4.864608, 0.010913803, 1.8637965, 2.8884032, 0.02226592, 1.969871, 2.3801248, 0, 2.2753072, 0.006579196, 1.9405862, 1.9717528, 2.623247, 0.0046527097, 1.536084, 1.2834355, 1.2332652, 1.4219276, 0.0073629213, 1.305458, 2.2002983, 2.1147442, 1.6076756, 1.5297375, 0.0044696387, 1.7108474, 1.1384524, 0.91492224, 1.419768, 0.24807997, 0.0038625724, 0.13964018, 0.10479091, 0.10267623, 0.006808863, 0.028114961, 0, 0.041466415, 0.010537256, 0.094644606, 0.021485792, 0.046756607, 0.09667912, 0.10854383, 0.052452713, 0.05416489, 0.01709799, 0.038735736, 0.00022084083, 0.034080032, 0.050248113, 0.07302172, 0.10474513, 0.03440547, 0.10824477, 0.08417762, 0.074672304, 0.028188596, 0.0616853, 0.06734598, 0.044802245, 0.06897127, 0, 0, 0.0118226865, 0.010032285, 0.02392773, 0.055860594, 0.1937616, 0.40691832, 0.33787966, 0.03786607, 0.13989739, 0.16260836, 0.114367954, 0.37490347, 0.5642893, 0.4508925, 0.713357, 1.0207783, 0.7616291, 0.9957552, 0.68612474, 1.1389803, 1.2752962, 0.903376, 1.5824292, 2.9347217, 3.5003483, 1.9656528, 2.8279176, 2.6326034, 2.6730273, 1.4753586, 1.8129238, 1.5784818, 1.2102742, 2.03812, 2.4216142, 1.3038746, 2.4912782, 2.186803, 1.3465528, 2.4821947, 2.6735122, 1.9106901, 1.7468477, 1.9028505, 2.9162683, 2.1738224, 2.079999, 2.6814466, 1.4720219, 0.09729797, 0.07791734, 0.10360346, 0.08646249, 0.08563267, 0.12517577, 0.07740597, 0.092006624, 0.054515243, 0.058936603, 0.19227149, 0.20707947, 0.15921754, 0.021557607, 0.0762301, 0.13540839, 0.027672632, 0.045735564, 0.09100423, 0.05944485, 0.057024807, 0.0854343, 0.035193995, 0.032887053, 0.03826888, 0.04199652, 0.05926308, 0.07320017, 0.07906411, 0.028566407, 0.07868506, 0.0222065, 0.030356297, 0.047669567, 0.06187102, 0.09852577, 0.045150153, 0.066567086, 0.064824775, 0.1368214, 0.04351154, 0.046954643, 0.054101814, 0.04980387, 0.12869896, 0.06435144, 0.05647961, 0.06574126, 0.039036214, 0.06544192, 0.044990342, 0.060292277, 0.042685714, 0.13062344, 0.09723069, 0.15987715, 0.14102256, 0.08333314, 0.062489428, 0.03767671, 0.02239845, 0.17567962, 0.30691764, 0.028768912, 0.26849133, 0.07036147, 0.04923495, 0.036997203, 0.07383414, 0.20634027, 0.21168658, 0.093210876, 0.071348995, 0.11422376, 0.117519476, 0.23806831, 0.12398842, 0.24876566, 0.09071729, 0.069282785, 0.11915278, 0.10840131, 0.07390671, 0.035632413, 0.14738369, 0.15971555, 0.09436129, 0.03428589, 0.10909621, 0.12402983, 0.06830104]
    mean_motion_03 = [0.11143272, 0.051823024, 0.052923277, 0.15620446, 0.06463343, 0.18100537, 0.075317845, 0.21810685, 0.08585137, 0.19018424, 0.17171782, 0.14010452, 0.1532764, 0.16176549, 0.1873678, 0.21267569, 0.2127149, 0.17444931, 0.15665035, 0.19408435, 0.17006934, 0.18449476, 0.17726843, 0.18885519, 0.19096808, 0.17017362, 0.1787132, 0, 0.22207229, 0.15240961, 0.18242995, 0.14928944, 0.17946124, 0.14248233, 0, 0, 0.1623384, 0.21723463, 0.068688266, 0.08332677, 0.12010358, 0.089270815, 0.059255805, 0.052570187, 0.045853317, 0.061501227, 0.07348361, 0.13423745, 0, 0, 0, 0, 1.5304252, 1.0401828, 1.4315077, 0, 0, 1.7421, 1.7930317, 1.6031277, 0, 1.3571877, 1.7077451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45937175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3695863, 2.7442935, 2.8414764, 3.6320233, 2.2334173, 0, 1.3483535, 1.363826, 1.5136849, 1.5915933, 1.6209904, 1.2003636, 1.196032, 0.8501653, 0.71089053, 0.5933803, 0.54766196, 0.828028, 1.895128, 1.9883267, 0.6472555, 0.96851015, 1.0103028, 0.1610608, 0.2566532, 0.14029117, 0.1368204, 0.13706148, 0.1285784, 0.022076229, 0.021524893, 0.019984381, 0.020381074, 0.023602068, 0.020470466, 0.021364655, 0.02209889, 0.16631146, 0.15538682, 0.15971884, 0.16183312, 0.018219426, 0.019987822, 0.011395652, 0.14378133, 0.02839901, 0.15905344, 0.16426279, 0.17755298, 0.17472228, 0.179177, 0.17792706, 0.15036714, 0.22931653, 0.012092112, 0.013672145, 0.008524105, 0.0073245447, 0.006577904, 0.016057428, 0.018009624, 0.007538564, 0.005681159, 0.005477822, 0.004934396, 0.01293462, 0.17335562, 0.18150626, 0.01842322, 0.023125168]
    mean_motion_05 = [0.20072924, 0.3056139, 0.3563416, 0.37115443, 0.16053359, 0.25051787, 0.22010037, 0.21837363, 0.18222845, 0.28085372, 0.27484888, 0.2803739, 0.31063837, 0.3188083, 0.19193286, 0.3888514, 0.42914143, 0.35953346, 0.42707288, 0.3458367, 0.43603003, 0.34338084, 0.33650434, 0.37317637, 0.28631678, 0.16818298, 0.13390946, 0.14557031, 0.14730167, 0.09731615, 0.11818148, 0.12839125, 0.142021, 0.13045995, 0.16126493, 0.16543168, 0.3375945, 0.19111088, 0.303549, 0.37877586, 0.5470454, 0.6841294, 0.5609432, 0.6143064, 0.37148583, 0.57199407, 0.49059197, 0.39156163, 0.5645539, 0.585539, 0.7299337, 0.53630996, 0.53164965, 0.31952462, 0.23201434, 0.24801677, 0.27011386, 0.2935197, 0.3050003, 0.28095904, 0.55935466, 0.46837994, 0.8112773, 0.29818687, 0.592263, 0.3647741, 0.45862368, 0.4007457, 0.37210727, 0.38914648, 0.58804756, 0.3993349, 0.3302141, 0.58733165, 0.5456949, 0.55027825, 0.53182113, 0.39838606, 0.28197607, 0.56373334, 0.15827867, 0.24523592, 0.10137248, 0.07977175, 0.0327017, 0.10132357, 0.14664353, 0.10936489, 0.067466475, 0.03845194, 0.14927204, 0.22710545, 0.3741612, 0.80544966, 1.2904805, 0.8918847, 0.6377834, 0.83951426, 1.1147377, 1.0586642, 1.066891, 1.2005074, 1.1648825, 1.0060887, 1.1303269, 0.79521877, 0.78572744, 0.5334445, 0.3749517, 0.6080979, 0.49970528, 0.68602544, 0.6294795, 0.37832108, 0.62740666, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.54283, 6.177227, 2.7940757, 1.431027, 1.4973359, 0.51144224, 1.100815, 1.1236608, 1.5105566, 1.6021143, 1.5734252, 1.4228206, 1.8530287, 1.9713584, 1.7298095, 1.4361799, 1.1716886, 0.71268266, 0.5428361, 0.550462, 0.6186613, 0.8663706, 2.092917, 2.509689, 0.6832376, 0.32438928, 0.21306658, 0.15476978, 0.118175395, 0.10641458, 0.10333355, 0.09932498, 0.079217136, 0.09535984, 0.0819249, 0.12639013, 0.15361398, 0.2028169, 0.1628179, 0.15339088, 0.122466825, 0.162456, 0.22704503, 0.16745651, 0.11619715, 0.10834543, 0.31203166, 0.19984551, 0.20236193, 0.16002773, 0.22897421, 0.09051695, 0.090076864, 0.07857535, 0.06407463, 0.044895366, 0.04719087, 0.058530834, 0.05920829, 0.058130108, 0.07470556, 0.09004826, 0.0632516, 0.06413886, 0.04160846, 0.054162003, 0.05006198, 0.06859884, 0.04791045, 0.045139868, 0.054511845, 0.040171195, 0.04046169, 0.03816888, 0.035868656, 0.017574858, 0.022159962, 0.032356832, 0.031256523, 0.031420134, 0.03155744, 0.0321119, 0.031846356, 0.029904049, 0.03479127, 0.038257264, 0.042088915, 0.048220742, 0.047961313, 0.028981678, 0.059482325, 0.042500127, 0.036946673, 0.01963963, 0.033184435, 0.093665354, 0.10326143, 0.18182714, 0.1521721, 0.13312076, 0.08131686]
    mean_motion_07 = [0.059062928, 0.0364131, 0.031560265, 0.03272246, 0.034869637, 0.028971856, 0.022718258, 0.028300019, 0.022815753, 0.022439579, 0.025488773, 0.05448019, 0.03550958, 0.06154553, 0.023586731, 0.036771476, 0.02609372, 0.045275908, 0.023690028, 0.030539233, 0.019328006, 0.017957095, 0.02034066, 0.032322064, 0.022232354, 0.035159945, 0.0271825, 0.024591457, 0.029065518, 0.027606824, 0.011859309, 0.02699886, 0.020902077, 0.020257715, 0.020701496, 0.027228018, 0.0440222, 0.028084204, 0.046634156, 0.014776577, 0.035900734, 0.02577627, 0.026466426, 0.027217275, 0.03175659, 0.026502008, 0.03365646, 0.050140966, 0.029454397, 0.03309542, 0.034506436, 0.025411429, 0.052406788, 0.013932062, 0.029587895, 0.027883388, 0.022886174, 0.02448927, 0.01548324, 0.051667947, 0.02722493, 0.029112572, 0.024805961, 0.014495669, 0.0394861, 0.027336756, 0.034962624, 0.031242289, 0.021253653, 0.030048832, 0.16308007, 0.29757318, 0.50081426, 0.56388324, 0.26679158, 0.12852065, 1.6073991, 3.3849535, 3.3447895, 2.6261728, 2.178899, 2.3200567, 0, 0, 4.645717, 0, 0, 0, 0, 0, 0.044984926, 4.948448, 3.1984904, 3.8108776, 2.8693807, 2.081858, 1.8575847, 1.392079, 2.702507, 1.7146302, 1.2843479, 0.7457814, 0.12710328, 0.28065762, 0.20926477, 0.20523116, 0.1580307, 0.12846571, 0.1870924, 0.348637, 0.43651485, 0.8818058, 0.69821495, 0.46143872, 0.40855587, 1.0757916, 0.48649395, 0.44959182, 0.17735526, 0.122641325, 0.16725564, 0.10304956, 0.05011438, 0.05558662, 0.064638875, 0.06304613, 0.05012175, 0.03777922, 0.041289393, 0.044604093, 0.053466357, 0.04180549, 0.031198923, 0.053838953, 0.040703688, 0.05199344, 0.037664395, 0.09537014, 0.030022789, 0.04582017, 0.025052208, 0.024567917, 0.03991255, 0.060907647, 0.040983222, 0.03710512, 0.040741127, 0.057543345, 0.02836441, 0.021954885, 0.040177293, 0.04361958, 0.03378805, 0.05795028, 0.17671454, 0.07219174, 0.095372975, 0.16105452, 0.108380124, 0.2502144, 0.43477836, 0.8900333, 1.0743226, 1.7206478, 1.6119636, 2.298854, 2.2467813, 1.8708274, 2.481037, 3.4684696, 3.394884, 3.5513105, 2.5844796, 2.4587238, 3.6090207, 3.8629153, 5.0479093, 5.575604, 4.8813305, 4.9259095, 4.024004, 4.9120865, 4.766465, 2.316723, 1.9687687, 5.3919263, 3.49325, 1.3435004, 1.6305565, 1.2107371, 0.8076671, 0.091486044, 0.031972755, 0.12152288, 0.0344979, 0.027790893, 0.077658534, 0.020007413, 0.018249972, 0.041514304, 0.016245235, 0.024867667, 0.03672035, 0.04215759, 0.02271953, 0.019429214, 0.030524017, 0.03624166, 0.018478366, 0.020056214, 0.01863712, 0.034687713, 0.01754862, 0.03092247, 0.025674846, 0.015830163, 0.016024914, 0.014122825, 0.018116744, 0.02461442, 0.019341405, 0.03948843, 0.022386435, 0.01726286, 0.010514708, 0.012450268, 0.018149681, 0.013609206, 0.027377535, 0.011777801, 0.022824315, 0.0136314565, 0.015178882, 0.020766383, 0.040550835, 0.020694125, 0.01882259, 0.011024559, 0.017964669, 0.01832892, 0.019573454, 0.016607791, 0.015251175, 0.03278209, 0.014647083, 0.019646721, 0.03654206, 0.023511365, 0.033033617, 0.027317638, 0.013223271]
    mean_motion_09 = [0.031069526, 0.035159584, 0.04934099, 0.035243526, 0.025088428, 0.011544723, 0.01897954, 0.04405377, 0.050183866, 0.040350303, 0.029353008, 0.040104553, 0.06447611, 0.08127646, 0.1471676, 0.13063407, 0.121951595, 0.13086747, 0.0961251, 0.122317314, 0.12747224, 0.027980868, 0.03377665, 0.048512142, 0.043342475, 0.041697994, 0.034588993, 0.03291684, 0.0312495, 0.028327756, 0.024029683, 0.018718602, 0.01363019, 0.010660363, 0.014815469, 0.010752903, 0.008933928, 0.010133528, 0.014551163, 0.012573317, 0.0077394857, 0.012522728, 0.003008981, 0.0077255666, 0.0056512537, 0.014770804, 0.0055766404, 0.00450027, 0.008155143, 0.0053750547, 0.13753399, 0.020659912, 0.010092791, 0.0079301605, 0.0052016294, 0.022270419, 0.12021111, 0.36527067, 0.27988416, 0.2727374, 0.24235535, 0.23236255, 0.46439308, 0.23104122, 0.5587219, 0.47732082, 0.50059396, 0.8575439, 1.6640153, 1.056028, 0.221294, 0.18218316, 1.0161316, 0.25239196, 0.13957588, 0.13793124, 0.15197828, 0.09714162, 0.11302457, 0.11737343, 0.14682363, 1.0902704, 0.5791524, 1.1112118, 0.30813783, 0.9745511, 0.9442774, 0.2541858, 0, 0, 0, 0, 0, 0.053485185, 0, 0.6613732, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5505013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8464907, 0, 0.91820294, 0.6700467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95880514, 0.34806228, 0.7203795, 0.32992837, 0.14538816, 0.167178, 0, 0, 0, 0, 0.15493923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20725702, 0.08620341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.059891853, 0.19217955, 0.24931109, 0.28630653, 0.44644177, 0.34335947, 0.24305655, 0.35772046, 0.54319876, 0, 1.3570372, 1.0365939, 1.2793282, 1.2002525, 0.3094236, 0.05101015, 0.017532358, 0.06559852]
       
    # mean_motion_07 = [0.061371535, 0.031981345, 0.024827769, 0.02502818, 0.031210128, 0.020289218, 0.015518503, 0.02019433, 0.015112214, 0.016386174, 0.017656567, 0.05556255, 0.029480847, 0.060390733, 0.01903133, 0.031489864, 0.01961706, 0.0436341, 0.014541645, 0.026074715, 0.013058673, 0.012450672, 0.012003143, 0.024107566, 0.014865308, 0.026423203, 0.019778308, 0.015978243, 0.022765428, 0.017964302, 0.0055426885, 0.019177644, 0.013082351, 0.01379713, 0.010057628, 0.017546667, 0.03987067, 0.021295682, 0.042073335, 0.00928046, 0.031109393, 0.018901471, 0.021122715, 0.02101914, 0.03093291, 0.017992161, 0.025069233, 0.04704405, 0.02144227, 0.030565996, 0.030478252, 0.017633105, 0.05082432, 0.009636681, 0.022817835, 0.021714699, 0.013359396, 0.01704726, 0.008896723, 0.04802203, 0.02081642, 0.02348737, 0.01891215, 0.008639384, 0.03570052, 0.02070686, 0.03022993, 0.027180586, 0.0145919975, 0.023419794, 0.16052982, 0.29610965, 0.4910236, 0.58785486, 0.27406856, 0.12636127, 1.6486447, 4.093678, 6.0541553, 6.9617405, 6.5384293, 5.644711, 6.127311, 6.9885187, 4.9286985, 3.9065914, 4.1590753, 1.7402025, 2.2190573, 4.062108, 0.00033579028, 5.37523, 3.1919115, 2.6327345, 2.1635766, 1.767738, 1.918913, 1.1469477, 2.7126439, 1.714112, 1.2875115, 0.74644893, 0.11173803, 0.27083552, 0.20951658, 0.21354352, 0.14698607, 0.11377775, 0.17650478, 0.33792964, 0.43875718, 0.8859235, 0.58167315, 0.42539445, 0.41382784, 1.0258439, 0.52453345, 0.44924933, 0.1349412, 0.10332421, 0.12569247, 0.07243473, 0.04511687, 0.034902774, 0.051613353, 0.042715997, 0.025867064, 0.027055424, 0.02160146, 0.011767357, 0.018152297, 0.030286122, 0.023211543, 0.047003925, 0.031991154, 0.037664425, 0.023093224, 0.029807271, 8.550005e-05, 0.0124192415, 0.018562287, 0.012462001, 0.026805913, 0.027176723, 0.019120956, 0.031067794, 0.037259497, 0.03248542, 0.023836162, 0.016329166, 0.0214911, 0.024616672, 0.017398302, 0.04068171, 0.16888838, 0.06299903, 0.0845988, 0.13670012, 0.08015218, 0.23699296, 0.43933454, 0.8566921, 1.0590163, 1.7089555, 1.5962543, 1.8984514, 2.1448271, 1.8481069, 2.3705213, 3.2827542, 3.1422331, 3.772603, 3.8413477, 2.4820492, 3.0525148, 3.5799065, 3.8588526, 4.112193, 3.9101741, 3.9207718, 5.18213, 6.7708826, 0, 4.085917, 4.084531, 5.509695, 3.5154934, 3.8540637, 3.6330802, 1.2401137, 0.8014099, 0.093002304, 0.02576473, 0.12178824, 0.030089257, 0.023934929, 0.07692029, 0.013391925, 0.0072345603, 0.034132127, 0.010449057, 0.01774633, 0.020838695, 0.029772934, 0.015727742, 0.013839471, 0.022967624, 0.0285006, 0.011655296, 0.012697802, 0.012056402, 0.026092075, 0.010081241, 0.024125742, 0.017899143, 0.0075453357, 0.009172091, 0.009501407, 0.010666465, 0.015604519, 0.011590683, 0.03052801, 0.0130124735, 0.012654735, 0.006316779, 0.0069388044, 0.010947775, 0.008605496, 0.022282565, 0.0064201104, 0.014639614, 0.009277006, 0.009964823, 0.015755292, 0.03417353, 0.014891959, 0.0150684295, 0.0065501174, 0.010077638, 0.012048846, 0.013495672, 0.011263096, 0.0091119595, 0.02304643, 0.010268106, 0.012295117, 0.031301443, 0.016696898, 0.027927458, 0.018828984, 0.0074950964]
    # mean_motion_09 = [0.048600405, 0.067943975, 0.08178723, 0.1074773, 0.03426945, 0.18026116, 0.10716559, 0.04026903, 0.183622, 0.052751847, 0.0634891, 0.08525934, 0.14357245, 0.12561782, 0.14125745, 0.09695394, 0.062134907, 0.060292803, 0.046557833, 0.10127055, 0.030749615, 0.0230101, 0.028359173, 0.04165483, 0.032877274, 0.030560715, 0.021731358, 0.022055738, 0.025984766, 0.025743265, 0.021266203, 0.02019806, 0.014056764, 0.0129440725, 0.015763957, 0.10699322, 0.08135056, 0.111054584, 0.09105308, 0.11689729, 0.10461366, 0.06293654, 0.0645553, 0.07709559, 0.087509036, 0.068481565, 0.051453467, 0.09107722, 0.08720167, 0.03654407, 0.16436413, 0.28795356, 0.25659078, 0.34153458, 0.28128496, 0.31313318, 0.381006, 0.70135933, 0.49198696, 0.21462952, 0.47894448, 0.49223894, 0.5297756, 0.24792458, 0.35953712, 0.62272316, 0.824195, 2.1892853, 0.7463188, 0.35762802, 0.25548735, 0.21335907, 0.28555205, 0.16244279, 0.14367908, 0.14443946, 0.122697555, 0.098454565, 0, 0, 1.2760522, 1.1295142, 1.0534418, 1.02827, 0.8683088, 1.1019832, 1.1169758, 1.0699561, 0.9139529, 0.83911866, 0.16234753, 0.10659335, 0.8663363, 0.7874164, 0.721687, 0.70957476, 0.73004407, 0.69101167, 0.69765234, 0.56721663, 0.6003193, 0.63110703, 0.6825612, 0.8316275, 0.71311903, 0.6559222, 0.6444851, 0.7327359, 0.6638107, 0.71085757, 0.69180745, 0.6905756, 0.6894774, 0.7293191, 0.77625775, 0.73081267, 0.6571856, 0.7701278, 0.7016524, 0.7318528, 0.6995566, 0.78257865, 0.79458123, 0.8055077, 0.7533274, 0.742014, 0.7379148, 0.6955269, 0.7471632, 0.78202975, 0.75923544, 0.7324951, 0.7843653, 0.7837566, 0.6667706, 0.67107576, 0.63093245, 0.68279403, 0.6603524, 0.7226066, 0.7478206, 0.8142309, 0.7426704, 0.6705854, 0.69278544, 0.6763794, 0.64370954, 0.7550156, 0.7206731, 0.717906, 0.75783527, 0.6824041, 0.7153202, 0.72064954, 0.6935297, 0.73558, 0.77832997, 0.6861835, 0.72585034, 0.72612417, 0.694025, 0.68740004, 0.7058246, 0.69543666, 0.70340365, 0.7578144, 0.739103, 0.7821628, 0.78940684, 0.7261525, 0.73971856, 0.7650406, 0.7162578, 0.78845465, 0.75709563, 0.7382258, 0.7394731, 0.7847976, 0.77923334, 0.7732335, 0.77715504, 0.78226113, 0.7264234, 0.72400343, 0.78179234, 0.76170754, 0.7440696, 0.7016717, 0.7719529, 0.77837914, 0.81546307, 0.7835052, 0.75276965, 0.6725628, 0.6946207, 0.73575807, 0.7626582, 0.77512205, 0.6966123, 0.71878314, 0.8018675, 0.8335628, 0.8346413, 0.8261823, 0.87473726, 0.70610905, 0.77986157, 0.7545619, 0.69850594, 0.5674035, 0.3812914, 0.35225052, 0.45010817, 0.42317012, 0.40909192, 0.4690979, 0.3586454, 0.49207398, 0.44893444, 0.41136822, 0.33325887, 0.41647235, 0.5046224, 0.50454473, 0.41051546, 0.4076337, 0.5043948, 0.514877, 0.38572678, 0.45311326, 0.49155012, 0.4352164, 0.5004014, 0.53380686, 0.47649965, 0.4826165, 0.3479377, 0.5323372, 0.48592004, 0.59550595, 0.45371634, 0.45889014, 0.47059393, 0.5118702, 0.61146873, 0.46549165, 0.47681195, 0.5395258, 0.63846344, 0, 0.16490646, 0.36645088, 0.37600845, 0, 0.4460932, 0.44651952, 0.42515448, 0.6122412, 0.40325865, 0.45505607, 0.41425568, 0.50843287, 0.5497252, 0, 0, 0, 0, 0.34214818, 0.4768604, 0, 0.62639326, 0.5688367, 0.55003864, 0.6054856, 0.57587874, 0.42002317, 0.41659486, 0.37397355, 0.3571025, 0, 0.3166672, 0.48500624, 0.5429692, 0.54657865, 0.59221345, 0.4784459, 0.5453076, 0.3651305, 0.48406053, 0.6216852, 0.42542067, 0.38325357, 0.28210127, 0.3150911, 0.4180117, 0.35007796, 0.38730115, 0.33302394, 0.46272472, 0.3788013, 0.38344836, 0.39987123, 0.38471153, 0.3173733, 0.33393624, 0.2893337, 0.40258527, 0.34854218, 0.3926825, 0.39050916, 0.4899127, 0.4422905, 0.4321624, 0.43663645, 0.37584695, 0.42130718, 0.3936237, 0.47937232, 0.42299625, 0.39028323, 0.39657223, 0.39197224, 0.4262339, 0.48154816, 0.4891913, 0.5097457, 0.41929293, 0.43398213, 0.4858389, 0.4873367, 0.34887332, 0.59378415, 0.87345296, 0.44268516, 1.5332725, 1.0954201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.6298106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.1175065, 1.5197768, 0, 0, 0, 0, 0, 0, 0, 0.51610154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.070318066, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.015710026, 0.006484301, 0.013861881, 0.022490684, 0.011943472, 0.01826191, 0.016317392, 0, 0.012713396, 0.018418519, 0, 0, 0, 0, 0.021892484, 0, 0, 0, 0, 0, 0, 0, 0, 0.017130079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5929727, 1.4796356, 1.4539845, 1.3515536, 0.90093297, 0.5051082, 0.048223983, 0.031611983, 0.09811647]
    # p = get_guessed_frame_list(mean_motion_01)
    # print("01: ", p, p[0] >= 80 and p[0] <= 98, p[1] >= 166 and p[1] <= 200) # 80 ~ 98 (116) 166 ~ 200
    # p = get_guessed_frame_list(mean_motion_03)
    # print("03: ", p, p[0] >= 50 and p[0] <= 65, p[1] >= 252 and p[1] <= 288) # 50 ~ 65 (80) 252 ~ 288
    # p = get_guessed_frame_list(mean_motion_05)
    # print("05: ", p, p[0] >= 93 and p[0] <= 110, p[1] >= 346 and p[1] <= 380) # 93 ~ 110 (127) 346 ~ 380
    # p = get_guessed_frame_list(mean_motion_07)
    # print("07: ", p, p[0] >= 76 and p[0] <= 90, p[1] >= 163 and p[1] <= 191) # 76 ~ 90 (105) 163 ~ 191
    # p = get_guessed_frame_list(mean_motion_09)
    # print("09: ", p, p[0] >= 60 and p[0] <= 75, p[1] >= 542 and p[1] <= 567) # 60 ~ 75 (90) 542 ~ 567

    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()
