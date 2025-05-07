from ultralytics import YOLO

def load_yolo_pose_model(weights):
    """
    Load the YOLO pose model with the specified weights.
    
    Args:
        weights (str): Path to the YOLO model weights file.
    
    Returns:
        YOLO: Initialized YOLO model object.
    """
    model = YOLO(weights)
    return model