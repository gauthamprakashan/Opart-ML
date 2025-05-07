import os
import cv2

def get_video_paths(video_folder):
    """
    Retrieve a list of video file paths from the specified folder.
    
    Args:
        video_folder (str): Path to the folder containing video files.
    
    Returns:
        list: List of video file paths.
    
    Raises:
        ValueError: If no videos are found in the folder.
    """
    video_extensions = (".mp4", ".avi", ".mov")
    video_paths = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(video_extensions)
    ]
    if not video_paths:
        raise ValueError(f"No videos found in folder: {video_folder}")
    return video_paths

def load_video(video_path):
    """
    Load a video file and return its capture object and properties.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        tuple: (VideoCapture object, frame_width, frame_height, fps)
    
    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, frame_width, frame_height, fps