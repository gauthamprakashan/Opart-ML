def log_actions(csv_file, frame_count, person_interactions, tracked_people):
    """
    Log detected actions to the CSV file.
    
    Args:
        csv_file (file object): Open CSV file in append mode.
        frame_count (int): Current frame number.
        person_interactions (defaultdict): Person ID to set of interacted object labels.
        tracked_people (dict): Tracked people data.
    """
    for person_id, interacted_objects in person_interactions.items():
        for obj_label in interacted_objects:
            action = f"P{person_id} working with {obj_label}"
            csv_line = f"{frame_count},{person_id},{action},{len(tracked_people)}\n"
            csv_file.write(csv_line)
            csv_file.flush()