import argparse
import os
import cv2
from src.data.video_loader import get_video_paths, load_video
from src.models.yolo_pose import load_yolo_pose_model
from src.tracking.tracker import ArmMovementTracker
from src.inference.action_logger import log_actions
from src.utils.helpers import calculate_iou_rotated, load_class_labels, load_object_coordinates, copy_to_dbfs
from src.utils.visualization import draw_annotations
from collections import defaultdict
import numpy as np

def main(args):
    """Process videos for pose tracking and action logging."""
    # Load class labels and object coordinates
    categories = load_class_labels(args.class_json)
    boxes = load_object_coordinates(args.cord_txt, categories)
    
    # Initialize tracker with Notebook-specified parameters
    movement_tracker = ArmMovementTracker(movement_threshold=7, frame_memory=15)
    
    # Load YOLO pose model
    pose_model = load_yolo_pose_model(args.model_weights)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Set up CSV file
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, "w") as f:
            f.write("frame_count,person_id,action,num_people\n")
    csv_file = open(args.csv_path, "a")
    
    # Get video paths
    video_paths = get_video_paths(args.video_folder)
    print(f"Found {len(video_paths)} videos: {video_paths}")
    
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        
        # Load video
        cap, frame_width, frame_height, fps = load_video(video_path)
        
        # Set up output video
        output_video_path = os.path.join(args.output_folder, f"output_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Process video
        frame_count = 0
        for pose_result in pose_model.track(video_path, stream=True, classes=[0], tracker="bytetrack.yaml"):
            frame_count += 1
            frame = pose_result.orig_img
            if frame_width == 0 or frame_height == 0:
                frame_height, frame_width = frame.shape[:2]
            
            # Process tracked people
            tracked_people = {}
            if pose_result.boxes is not None and pose_result.boxes.id is not None:
                for box, track_id, keypoints in zip(pose_result.boxes.xyxy.cpu().numpy(),
                                                  pose_result.boxes.id.cpu().numpy(),
                                                  pose_result.keypoints.data.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    tracked_people[int(track_id)] = {
                        'box': (x1, y1, x2, y2),
                        'center': center,
                        'keypoints': keypoints
                    }
                print(f"Frame {frame_count}: Tracked {len(tracked_people)} people")
            
            # Process poses
            poses = []
            for track_id, data in tracked_people.items():
                keypoints = data['keypoints']
                valid_points = keypoints[keypoints[:, 2] > 0.5, :2]
                if len(valid_points) > 0:
                    kp_center = np.mean(valid_points, axis=0)
                    poses.append({
                        'track_id': track_id,
                        'keypoints': keypoints,
                        'center': (kp_center[0], kp_center[1])
                    })
            print(f"Frame {frame_count}: Detected {len(poses)} poses")
            
            # Action detection and annotation
            current_pids = {}
            body_part_boxes = []
            person_interactions = defaultdict(set)
            pid = 0
            for pose in sorted(poses, key=lambda x: x['track_id']):
                track_id = pose['track_id']
                current_pids[pid] = {'track_id': track_id, 'pose': pose}
                keypoints = pose['keypoints']
                
                # Detect arm movements
                for arm_side, wrist_idx, elbow_idx in [('left', 9, 7), ('right', 10, 8)]:
                    if keypoints[wrist_idx][2] > 0.65 and keypoints[elbow_idx][2] > 0.65:
                        wrist = keypoints[wrist_idx][:2]
                        elbow = keypoints[elbow_idx][:2]
                        is_action_active = movement_tracker.update_and_check_movement(pid, arm_side, wrist, elbow, frame_count)
                        min_x = max(0, min(wrist[0], elbow[0]) - 20)
                        min_y = max(0, min(wrist[1], elbow[1]) - 20)
                        max_x = min(frame_width, max(wrist[0], elbow[0]) + 20)
                        max_y = min(frame_height, max(wrist[1], elbow[1]) + 20)
                        arm_points = [
                            (min_x / frame_width, min_y / frame_height),
                            (max_x / frame_width, min_y / frame_height),
                            (max_x / frame_width, max_y / frame_height),
                            (min_x / frame_width, max_y / frame_height)
                        ]
                        body_part_boxes.append({
                            'box': [min_x, min_y, max_x, max_y],
                            'points': arm_points,
                            'is_active': is_action_active,
                            'person_id': pid,
                            'hand': arm_side
                        })
                
                # Check object interactions
                for box in boxes:
                    for body_part in body_part_boxes:
                        if body_part['person_id'] == pid:
                            wrist_idx = 9 if body_part['hand'] == 'left' else 10
                            if keypoints[wrist_idx][2] > 0.5:
                                iou = calculate_iou_rotated(body_part["points"], box['points'], frame_width, frame_height)
                                if iou > 0.05 and body_part['is_active']:
                                    person_interactions[pid].add(box['label'])
                
                pid += 1
            
            # Log actions
            log_actions(csv_file, frame_count, person_interactions, tracked_people)
            
            # Draw annotations
            annotated_frame = draw_annotations(frame, tracked_people, poses, body_part_boxes, person_interactions, current_pids)
            
            # Write frame to output video
            out.write(annotated_frame)
        
        # Cleanup for this video
        cap.release()
        out.release()
        print(f"Finished processing and saved video: {output_video_path}")
    
    # Final cleanup
    csv_file.close()
    if args.dbfs_csv_path:
        copy_to_dbfs(f"file:{args.csv_path}", args.dbfs_csv_path)
    if args.print_csv:
        with open(args.csv_path, "r") as f:
            print("Contents of CSV:")
            print(f.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object tracking with pose estimation.")
    parser.add_argument("--video_folder", default="data/videos", help="Folder containing input videos")
    parser.add_argument("--output_folder", default="output", help="Folder for output videos")
    parser.add_argument("--csv_path", default="output/tracked_actions.csv", help="Path to output CSV file")
    parser.add_argument("--dbfs_csv_path", default=None, help="DBFS path to copy CSV to (Databricks only)")
    parser.add_argument("--class_json", default="data/class.json", help="Path to class labels JSON")
    parser.add_argument("--cord_txt", default="data/cord.txt", help="Path to object coordinates text file")
    parser.add_argument("--model_weights", default="yolo11n-pose.pt", help="Path to YOLO model weights")
    parser.add_argument("--print_csv", action="store_true", help="Print CSV contents after processing")
    args = parser.parse_args()
    
    main(args)