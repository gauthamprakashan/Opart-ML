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
        if len(history) >= 2:
            if self.check_significant_movement(history):
                tracker['movement_counter'][side] += 1
                if tracker['movement_counter'][side] >= self.frame_memory:
                    tracker['action_active'][side] = True
            else:
                tracker['movement_counter'][side] = 0
                tracker['action_active'][side] = False
        return tracker['action_active'][side]

class LegMovementTracker:
    def __init__(self, movement_threshold=0.01, frame_memory=15, min_walking_duration=0.5, fps=30):
        """
        Initialize the leg movement tracker for walking detection.
        
        Args:
            movement_threshold (float): Threshold for significant knee movement (fraction of frame height).
            frame_memory (int): Number of frames to keep in history.
            min_walking_duration (float): Minimum duration (seconds) to confirm walking.
            fps (int): Frames per second of the video.
        """
        self.trackers = {}
        self.movement_threshold = movement_threshold
        self.frame_memory = frame_memory
        self.min_walking_frames = int(min_walking_duration * fps)
        self.fps = fps

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
                'prev_keypoints': {'left': {'knee': None, 'ankle': None}, 'right': {'knee': None, 'ankle': None}},
                'keypoint_history': {'left': [], 'right': []},
                'walking_counter': 0,
                'is_walking': False,
                'last_logged_second': -1
            }
        return self.trackers[person_id]

    def calculate_keypoint_movement(self, person_id, side, knee_pos, ankle_pos, frame_count):
        """
        Update keypoint history for a leg side.
        
        Args:
            person_id (int): Person identifier.
            side (str): 'left' or 'right' leg.
            knee_pos (tuple): (x, y) coordinates of the knee.
            ankle_pos (tuple): (x, y) coordinates of the ankle.
            frame_count (int): Current frame number.
        
        Returns:
            list: Updated keypoint history.
        """
        tracker = self.get_tracker(person_id)
        prev_keypoints = tracker['prev_keypoints']
        history = tracker['keypoint_history'][side]
        if frame_count % 3 == 0:
            current_keypoints = {'knee': list(knee_pos), 'ankle': list(ankle_pos), 'frame': frame_count}
            history.append(current_keypoints)
            if len(history) > self.frame_memory:
                history.pop(0)
            prev_keypoints[side]['knee'] = list(knee_pos)
            prev_keypoints[side]['ankle'] = list(ankle_pos)
        return history

    def check_walking_pattern(self, left_history, right_history, person_id, frame_count, frame_height):
        """
        Check if knee movements indicate walking.
        
        Args:
            left_history (list): Keypoint history for left leg.
            right_history (list): Keypoint history for right leg.
            person_id (int): Person identifier.
            frame_count (int): Current frame number.
            frame_height (int): Height of the video frame for normalization.
        
        Returns:
            bool: True if walking pattern is detected.
        """
        if len(left_history) < 2 or len(right_history) < 2:
            return False
        total_movement = 0
        for history, side in [(left_history, 'left'), (right_history, 'right')]:
            knee_y = [h['knee'][1] / frame_height for h in history[-self.frame_memory:]]
            if len(knee_y) < 2:
                return False
            movement = sum(abs(knee_y[i] - knee_y[i-1]) for i in range(1, len(knee_y)))
            total_movement += movement / max(1, len(knee_y) - 1)  # Average per frame
        is_significant = total_movement > self.movement_threshold
        print(f"Frame {frame_count}, Person {person_id}: Total knee movement={total_movement:.4f}, Significant={is_significant}")
        return is_significant

    def update_and_check_walking(self, person_id, left_knee_pos, left_ankle_pos, right_knee_pos, right_ankle_pos, frame_count, frame_height):
        """
        Update tracker and check for walking.
        
        Args:
            person_id (int): Person identifier.
            left_knee_pos (tuple): Left knee coordinates.
            left_ankle_pos (tuple): Left ankle coordinates.
            right_knee_pos (tuple): Right knee coordinates.
            right_ankle_pos (tuple): Right ankle coordinates.
            frame_count (int): Current frame number.
            frame_height (int): Height of the video frame for normalization.
        
        Returns:
            tuple: (is_walking, should_log) - boolean indicating walking and whether to log.
        """
        tracker = self.get_tracker(person_id)
        left_history = self.calculate_keypoint_movement(person_id, 'left', left_knee_pos, left_ankle_pos, frame_count)
        right_history = self.calculate_keypoint_movement(person_id, 'right', right_knee_pos, right_ankle_pos, frame_count)
        is_walking = False
        if frame_count % 3 == 0 and len(left_history) >= 2 and len(right_history) >= 2:
            if self.check_walking_pattern(left_history, right_history, person_id, frame_count, frame_height):
                tracker['walking_counter'] += 1
                if tracker['walking_counter'] >= self.min_walking_frames:
                    is_walking = True
            else:
                tracker['walking_counter'] = max(0, tracker['walking_counter'] - 1)  # Decay
            tracker['is_walking'] = is_walking
            print(f"Frame {frame_count}, Person {person_id}: Walking={is_walking}, Counter={tracker['walking_counter']}")
        current_second = frame_count // self.fps
        should_log = is_walking and (current_second > tracker['last_logged_second'])
        if should_log:
            tracker['last_logged_second'] = current_second
            print(f"Frame {frame_count}, Person {person_id}: Should log walking")
        return is_walking, should_log