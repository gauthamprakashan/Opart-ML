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

class InferencePipeline:
    """Manages the inference process for pose tracking and action detection."""
    
    def __init__(self, args):
        self.args = args
        self.categories = self._load_class_labels()
        self.boxes = self._load_object_coordinates()
        self.movement_tracker = ArmMovementTracker(movement_threshold=7, frame_memory=15)
        self.pose_model = self._load_yolo_pose_model()
        self.csv_file = self._setup_csv()

    def _load_class_labels(self):
        """Load class labels from JSON file."""
        return load_class_labels(self.args.class_json)

    def _load_object_coordinates(self):
        """Load object coordinates from text file."""
        return load_object_coordinates(self.args.cord_txt, self.categories)

    def _load_yolo_pose_model(self):
        """Load YOLO pose model from weights."""
        return load_yolo_pose_model(self.args.model_weights)

    def _setup_csv(self):
        """Set up CSV file for action logging."""
        os.makedirs(os.path.dirname(self.args.csv_path), exist_ok=True)
        if not os.path.exists(self.args.csv_path):
            with open(self.args.csv_path, "w") as f:
                f.write("frame_count,person_id,action,num_people\n")
        return open(self.args.csv_path, "a")

    def run(self):
        """Execute the inference pipeline for all videos."""
        video_paths = get_video_paths(self.args.video_folder)
        print(f"Found {len(video_paths)} videos: {video_paths}")
        
        for video_path in video_paths:
            self._process_video(video_path)
        
        self.csv_file.close()
        if self.args.dbfs_csv_path:
            copy_to_dbfs(f"file:{self.args.csv_path}", self.args.dbfs_csv_path)
        if self.args.print_csv:
            with open(self.args.csv_path, "r") as f:
                print("Contents of CSV:")
                print(f.read())

    def _process_video(self, video_path):
        """Process a single video through the inference pipeline."""
        print(f"Processing video: {video_path}")
        
        cap, frame_width, frame_height, fps = load_video(video_path)

        if self.args.generate_video:
            output_path = os.path.join(self.args.output_folder, f"output_{os.path.basename(video_path)}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_count = 0
        for pose_result in self.pose_model.track(video_path, stream=True, classes=[0], tracker="bytetrack.yaml"):
            frame_count += 1
            frame = pose_result.orig_img
            frame_height, frame_width = frame.shape[:2] if frame_width == 0 or frame_height == 0 else (frame_height, frame_width)

            tracked_people = self._extract_tracked_people(pose_result, frame_count)
            poses = self._analyze_poses(tracked_people, frame_count)
            annotations = self._infer_actions(poses, frame_count, frame_width, frame_height)
            
            log_actions(self.csv_file, frame_count, annotations['person_interactions'], tracked_people)
            if self.args.generate_video:
                annotated_frame = draw_annotations(
                    frame, tracked_people, poses, annotations['body_part_boxes'], 
                    annotations['person_interactions'], annotations['current_pids'],
                    self.boxes,frame_width,frame_height
                )
                out.write(annotated_frame)

        cap.release()
        if self.args.generate_video:
            out.release()
            print(f"Finished processing and saved video: {output_path}")

    def _extract_tracked_people(self, pose_result, frame_count):
        """Extract tracked individuals from pose results."""
        tracked_people = {}
        if pose_result.boxes is None or pose_result.boxes.id is None:
            return tracked_people

        boxes = pose_result.boxes.xyxy.cpu().numpy()
        track_ids = pose_result.boxes.id.cpu().numpy()
        keypoints = pose_result.keypoints.data.cpu().numpy()

        tracked_people = {
            int(track_id): {
                'box': tuple(map(int, box)),
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                'keypoints': kp
            } for box, track_id, kp in zip(boxes, track_ids, keypoints)
        }
        
        print(f"Frame {frame_count}: Tracked {len(tracked_people)} people")
        return tracked_people

    def _analyze_poses(self, tracked_people, frame_count):
        """Analyze keypoints to identify valid poses."""
        poses = [
            {
                'track_id': track_id,
                'keypoints': data['keypoints'],
                'center': np.mean(data['keypoints'][data['keypoints'][:, 2] > 0.5, :2], axis=0)
            } for track_id, data in tracked_people.items()
            if len(data['keypoints'][data['keypoints'][:, 2] > 0.5, :2]) > 0
        ]
        
        print(f"Frame {frame_count}: Detected {len(poses)} poses")
        return poses

    def _infer_actions(self, poses, frame_count, frame_width, frame_height):
        """Infer actions and interactions from poses."""
        current_pids = {}
        body_part_boxes = []
        person_interactions = defaultdict(set)
        pid = 0

        for pose in sorted(poses, key=lambda x: x['track_id']):
            track_id = pose['track_id']
            current_pids[pid] = {'track_id': track_id, 'pose': pose}
            keypoints = pose['keypoints']

            for arm_side, wrist_idx, elbow_idx in [('left', 9, 7), ('right', 10, 8)]:
                if keypoints[wrist_idx][2] > 0.65 and keypoints[elbow_idx][2] > 0.65:
                    wrist = keypoints[wrist_idx][:2]
                    elbow = keypoints[elbow_idx][:2]
                    body_part_boxes.append(self._track_arm_movement(
                        pid, arm_side, wrist, elbow, frame_count, frame_width, frame_height
                    ))

            for box in self.boxes:
                for body_part in body_part_boxes:
                    if body_part['person_id'] == pid and body_part['is_active']:
                        wrist_idx = 9 if body_part['hand'] == 'left' else 10
                        if keypoints[wrist_idx][2] > 0.5:
                            iou = calculate_iou_rotated(body_part["points"], box['points'], frame_width, frame_height)
                            if iou > 0.05:
                                person_interactions[pid].add(box['label'])
            
            pid += 1

        return {
            'current_pids': current_pids,
            'body_part_boxes': body_part_boxes,
            'person_interactions': person_interactions
        }

    def _track_arm_movement(self, pid, arm_side, wrist, elbow, frame_count, frame_width, frame_height):
        """Track and analyze arm movement."""
        is_action_active = self.movement_tracker.update_and_check_movement(pid, arm_side, wrist, elbow, frame_count)
        min_x = max(0, min(wrist[0], elbow[0]) - 20)
        min_y = max(0, min(wrist[1], elbow[1]) - 20)
        max_x = min(frame_width, max(wrist[0], elbow[0]) + 20)
        max_y = min(frame_height, max(wrist[1], elbow[1]) + 20)
        
        return {
            'box': [min_x, min_y, max_x, max_y],
            'points': [
                (min_x / frame_width, min_y / frame_height),
                (max_x / frame_width, min_y / frame_height),
                (max_x / frame_width, max_y / frame_height),
                (min_x / frame_width, max_y / frame_height)
            ],
            'is_active': is_action_active,
            'person_id': pid,
            'hand': arm_side
        }

def main(args):
    """Initialize and run the inference pipeline."""
    pipeline = InferencePipeline(args)
    pipeline.run()

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
    parser.add_argument("--generate_video", action="store_true", default=False, help="Generate output video with annotated frames")    
    args = parser.parse_args()
    
    main(args)