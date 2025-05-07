import numpy as np
from collections import defaultdict

class ArmMovementTracker:
    def __init__(self, movement_threshold=13, frame_memory=15):
        """
        Initialize the arm movement tracker.
        
        Args:
            movement_threshold (float): Threshold for significant movement (pixels).
            frame_memory (int): Number of frames to keep in history.
        """
        self.trackers = {}
        self.movement_threshold = movement_threshold
        self.frame_memory = frame_memory

    def get_tracker(self, person_id):
        """
        Get or create a tracker for a specific person.
        
        Args:
            person_id (int): Unique identifier for the person.
        
        Returns:
            dict: Tracker data for the person.
        """
        if person_id not in self.trackers:
            self.trackers[person_id] = {
                'prev_keypoints': {'left': {'wrist': None, 'elbow': None}, 'right': {'wrist': None, 'elbow': None}},
                'movement_counter': {'left': 0, 'right': 0},
                'active_action': None,
                'keypoint_history': {'left': [], 'right': []},
                'action_active': {'left': False, 'right': False}
            }
        return self.trackers[person_id]

    def calculate_keypoint_movement(self, person_id, side, wrist_pos, elbow_pos, frame_count):
        """
        Update keypoint history for an arm side.
        
        Args:
            person_id (int): Person identifier.
            side (str): 'left' or 'right' arm.
            wrist_pos (tuple): (x, y) coordinates of the wrist.
            elbow_pos (tuple): (x, y) coordinates of the elbow.
            frame_count (int): Current frame number.
        
        Returns:
            list: Updated keypoint history.
        """
        tracker = self.get_tracker(person_id)
        prev_keypoints = tracker['prev_keypoints']
        history = tracker['keypoint_history'][side]
        
        if frame_count % 3 == 0:
            current_keypoints = {'wrist': wrist_pos, 'elbow': elbow_pos, 'frame': frame_count}
            history.append(current_keypoints)
            if len(history) > self.frame_memory:
                history.pop(0)
            prev_keypoints[side]['wrist'] = wrist_pos
            prev_keypoints[side]['elbow'] = elbow_pos
        return history

    def check_significant_movement(self, history):
        """
        Check if movement in history exceeds the threshold.
        
        Args:
            history (list): List of keypoint dictionaries.
        
        Returns:
            bool: True if significant movement detected.
        """
        if len(history) < 2:
            return False
        total_movement = 0
        for i in range(len(history) - 1):
            curr = history[i]
            next_frame = history[i + 1]
            wrist_movement = np.sqrt((curr['wrist'][0] - next_frame['wrist'][0])**2 + 
                                     (curr['wrist'][1] - next_frame['wrist'][1])**2)
            elbow_movement = np.sqrt((curr['elbow'][0] - next_frame['elbow'][0])**2 + 
                                     (curr['elbow'][1] - next_frame['elbow'][1])**2)
            total_movement += (wrist_movement + elbow_movement) / 2
        avg_movement = total_movement / (len(history) - 1)
        return avg_movement > self.movement_threshold

    def update_and_check_movement(self, person_id, side, wrist_pos, elbow_pos, frame_count):
        """
        Update tracker and check for significant movement.
        
        Args:
            person_id (int): Person identifier.
            side (str): 'left' or 'right' arm.
            wrist_pos (tuple): Wrist coordinates.
            elbow_pos (tuple): Elbow coordinates.
            frame_count (int): Current frame number.
        
        Returns:
            bool: True if action is active.
        """
        tracker = self.get_tracker(person_id)
        history = self.calculate_keypoint_movement(person_id, side, wrist_pos, elbow_pos, frame_count)
        if frame_count % 3 == 0 and len(history) >= 2:
            if self.check_significant_movement(history):
                tracker['movement_counter'][side] += 1
                if tracker['movement_counter'][side] >= self.frame_memory:
                    tracker['action_active'][side] = True
            else:
                tracker['movement_counter'][side] = 0
                tracker['action_active'][side] = False
        return tracker['action_active'][side]