import numpy as np
import cv2
import json
import os

def calculate_iou_rotated(points1, points2, frame_width, frame_height):
    """
    Calculate Intersection over Union for rotated polygons.
    
    Args:
        points1 (list): List of (x, y) coordinates for the first polygon.
        points2 (list): List of (x, y) coordinates for the second polygon.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
    
    Returns:
        float: IoU value.
    """
    pts1 = np.array(points1, dtype=np.float32)
    pts2 = np.array(points2, dtype=np.float32)
    x_min = min(np.min(pts1[:, 0]), np.min(pts2[:, 0]))
    y_min = min(np.min(pts1[:, 1]), np.min(pts2[:, 1]))
    x_max = max(np.max(pts1[:, 0]), np.max(pts2[:, 0]))
    y_max = max(np.max(pts1[:, 1]), np.max(pts2[:, 1]))
    w = int((x_max - x_min) * frame_width)
    h = int((y_max - y_min) * frame_height)
    if w <= 0 or h <= 0:
        return 0.0
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    pts1_mask = (pts1 - [x_min, y_min]) * [w, h]
    pts2_mask = (pts2 - [x_min, y_min]) * [w, h]
    cv2.fillPoly(mask1, [pts1_mask.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [pts2_mask.astype(np.int32)], 1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def load_class_labels(class_json):
    """
    Load class labels from a JSON file.
    
    Args:
        class_json (str): Path to the class JSON file.
    
    Returns:
        dict: Mapping of class IDs to names.
    """
    with open(class_json, 'r') as f:
        class_data = json.load(f)
        categories = {cat['id']: cat['name'] for cat in class_data['categories']}
    return categories

def load_object_coordinates(cord_txt, categories):
    """
    Load object coordinates from a text file.
    
    Args:
        cord_txt (str): Path to the coordinates text file.
        categories (dict): Class ID to name mapping.
    
    Returns:
        list: List of object box dictionaries.
    """
    boxes = []
    with open(cord_txt, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])
            points = [(values[i], values[i + 1]) for i in range(1, len(values)-1, 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_center = sum(xs) / len(xs)
            y_center = sum(ys) / len(ys)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            boxes.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'points': points,
                'label': categories.get(class_id, f"Class_{class_id}")
            })
    return boxes

def copy_to_dbfs(local_path, dbfs_path):
    """
    Copy a local file to DBFS if running in Databricks.
    
    Args:
        local_path (str): Local file path (e.g., "file:/path").
        dbfs_path (str): DBFS destination path (e.g., "dbfs:/path").
    """
    try:
        from databricks import dbutils
        dbutils.fs.cp(local_path, dbfs_path)
        print(f"Copied {local_path} to {dbfs_path}")
    except ImportError:
        print("Not in Databricks environment - keeping local file at", local_path.replace("file:", ""))