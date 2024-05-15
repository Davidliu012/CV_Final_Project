import json

def load_json(filename):
    """ Load JSON file from the specified filename. """
    with open(filename, 'r') as file:
        return json.load(file)

def find_matching_video(ground_truth_videos, video_filename):
    """ Find the matching video in ground truth data based on filename. """
    for video in ground_truth_videos:
        if video['video_filename'] == video_filename:
            return video
    return None

def evaluate_video(video_data, ground_truth_data):
    """ Evaluate a single video's data against the ground truth. """
    score = 0
    total_checks = len(ground_truth_data['states'])

    for gt_annotation in ground_truth_data['states']:
        annotation = next((state for state in video_data['states'] if state['state_id'] == gt_annotation['state_id']), None)
        if annotation:
            if gt_annotation['description'] == "Opening":
                if gt_annotation['start_frame'] <= annotation.get('guessed_frame', -1) <= gt_annotation['half_open_frame']:
                    score += 1
            elif gt_annotation['description'] == "Closing":
                if gt_annotation['start_frame'] <= annotation.get('guessed_frame', -1) <= gt_annotation['end_frame']:
                    score += 1
        # No score added if no annotation found
    return score, total_checks

def evaluate_algorithm(output_json, ground_truth_json):
    """ Evaluate algorithm output against ground truth annotations. """
    algorithm_data = load_json(output_json)
    ground_truth_data = load_json(ground_truth_json)

    total_score = 0
    total_possible = 0
    
    # Iterate through each video in the algorithm's output
    for algo_vid in algorithm_data['videos']:
        gt_vid = find_matching_video(ground_truth_data['videos'], algo_vid['video_filename'])
        if gt_vid:
            score, checks = evaluate_video(algo_vid['annotations'][0], gt_vid['annotations'][0])  # Assuming one annotation per video
            total_score += score
            total_possible += checks
        else:
            print(f"No ground truth data found for {algo_vid['video_filename']}")

    # Calculate final score as a percentage of the maximum possible score
    if total_possible == 0:
        return 0  # Avoid division by zero
    final_score = (total_score / total_possible) * 100
    return final_score

# Example usage
output_json_filename = 'algorithm_output.json'
ground_truth_json_filename = 'ground_truth_annotations.json'

final_score = evaluate_algorithm(output_json_filename, ground_truth_json_filename)
print(f"Final Score: {final_score:.2f} points")
