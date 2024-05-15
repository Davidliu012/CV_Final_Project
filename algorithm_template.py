import os
import json

def guess_door_opening(video_filename):
    """ Simulate guessing the frame for door opening. """
    # Hypothetical function: replace with actual logic.
    return 100  # Placeholder value, should be replaced with real computation based on video analysis.

def guess_door_closing(video_filename):
    """ Simulate guessing the frame for door closing. """
    # Hypothetical function: replace with actual logic.
    return 200  # Placeholder value, should be replaced with real computation based on video analysis.

def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    videos_info = []

    for video_file in video_files:
        videos_info.append({
            "video_filename": video_file,
            "annotations": [
                {
                    "object": "Door",
                    "states": [
                        {
                            "state_id": 1,
                            "description": "Opening",
                            "guessed_frame": guess_door_opening(video_file)  # Guessing frame using function.
                        },
                        {
                            "state_id": 2,
                            "description": "Closing",
                            "guessed_frame": guess_door_closing(video_file)  # Guessing frame using function.
                        }
                    ]
                }
            ]
        })

    return videos_info

def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def main():
    directory = "./"  # Specify the directory to scan
    output_filename = "output/algorithm_output.json"  # Output JSON file name
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()
