import cv2

def draw_annotations(frame, tracked_people, poses, body_part_boxes, person_interactions, current_pids,boxes,frame_width,frame_height):
    """
    Draw annotations on the video frame.
    
    Args:
        frame (np.ndarray): Video frame to annotate.
        tracked_people (dict): Tracked people data with boxes and keypoints.
        poses (list): Pose data with keypoints.
        body_part_boxes (list): Arm movement boxes.
        person_interactions (defaultdict): Person ID to interacted objects.
        current_pids (dict): Mapping of person IDs to track IDs and poses.
    
    Returns:
        np.ndarray: Annotated frame.
    """
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    # Draw tracked people boxes and IDs
    for track_id, data in tracked_people.items():
        x1, y1, x2, y2 = data['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 225, 109), 2)  # Teal #6DE1D2
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 225, 109), 2)  # Teal #6DE1D2

    # Draw keypoints
    track_id_to_pid = {current_pids[pid]['track_id']: pid for pid in current_pids}

    # Draw keypoints and check for valid poses
    for pose in poses:
        track_id = pose['track_id']
        keypoints = pose['keypoints']
        # Check if keypoints are valid
        has_valid_keypoints = keypoints is not None and len(keypoints) > 0 and keypoints.shape[1] > 0
        pid = track_id_to_pid.get(track_id)

        # Draw keypoints and skeleton if valid
        if has_valid_keypoints:
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (90, 90, 247), -1)  # Red #F75A5A
            for (start_idx, end_idx) in SKELETON:
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                if start_point[2] > 0.5 and end_point[2] > 0.5:
                    cv2.line(frame, (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])), (58, 214, 255), 2)  # Yellow #FFD63A
        # Annotate "no action detected" if no valid keypoints and person has no interactions
        elif pid is not None and pid not in person_interactions:
            if track_id in tracked_people:
                x1, y1, _, _ = tracked_people[track_id]['box']
                cv2.putText(frame, f"P{pid} no action detected", (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Draw arm boxes
    for body_part in body_part_boxes:
        min_x, min_y, max_x, max_y = body_part['box']
        color = (0, 255, 255) if body_part['is_active'] else (255, 0, 0)
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, 2)
    
    # Draw action text
    for person_id, interacted_objects in person_interactions.items():
        if person_id in current_pids:
            track_id = current_pids[person_id]['track_id']
            x1, y1, _, _ = tracked_people[track_id]['box']
            for obj_label in interacted_objects:
                action = f"P{person_id} working with {obj_label}"
                cv2.putText(frame, action, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw labelled kitchen object boxes

    # for box in boxes:
    #     points_abs = [(int(x * frame_width), int(y * frame_height)) for x, y in box['points']]
    #     for i in range(len(points_abs)):
    #         cv2.line(frame, points_abs[i], points_abs[(i + 1) % len(points_abs)], (0, 255, 0), 2)
    #     cv2.putText(frame, box['label'], points_abs[0], 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    
    return frame



    