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
    # Draw tracked people boxes and IDs
    for track_id, data in tracked_people.items():
        x1, y1, x2, y2 = data['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw keypoints
    for pose in poses:
        keypoints = pose['keypoints']
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    
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
    for box in boxes:
        points_abs = [(int(x * frame_width), int(y * frame_height)) for x, y in box['points']]
        for i in range(len(points_abs)):
            cv2.line(frame, points_abs[i], points_abs[(i + 1) % len(points_abs)], (0, 255, 0), 2)
        cv2.putText(frame, box['label'], points_abs[0], 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    
    return frame