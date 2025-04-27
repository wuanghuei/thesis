import torch

def debug_detection_stats(batch_detections, batch_size, metadata):
    """Print detection statistics for debugging"""
    total_dets = sum(len(dets) for dets in batch_detections)
    if total_dets == 0:
        print("WARNING: No detections in batch!")
        return
    
    print(f"Detections in batch: {total_dets} (avg {total_dets/batch_size:.1f} per sample)")
    
    # Count detections per class
    class_counts = {}
    for i, dets in enumerate(batch_detections):
        video_id = metadata[i]["video_id"] if i < len(metadata) else "unknown"
        print(f"Sample {i} (video {video_id}): {len(dets)} detections")
        
        for det in dets:
            action_id = det["action_id"]
            if action_id not in class_counts:
                class_counts[action_id] = 0
            class_counts[action_id] += 1
    
    # Print class statistics
    for action_id, count in sorted(class_counts.items()):
        print(f"  Class {action_id}: {count} detections")
    
    # Print detection details for first few detections
    if total_dets > 0:
        print("\nDetection details (first 3):")
        count = 0
        for i, dets in enumerate(batch_detections):
            if len(dets) > 0:
                for det in dets[:min(3, len(dets))]:
                    print(f"Det {count}: Class {det['action_id']}, Start: {det['start_frame']}, End: {det['end_frame']}, Conf: {det['confidence']:.4f}")
                    count += 1
                    if count >= 3:
                        break
            if count >= 3:
                break

def debug_raw_predictions(action_probs):
    """Analyze raw prediction values before thresholding"""
    # Check variance in predictions (helpful to detect potential collapse)
    action_variance = torch.var(action_probs).item()
    print(f"Action prediction variance: {action_variance:.6f}")
    
    # Check per-class stats
    for c in range(action_probs.shape[2]):  # For each class
        class_probs = action_probs[:, :, c]
        print(f"Class {c}: min={class_probs.min().item():.4f}, max={class_probs.max().item():.4f}, mean={class_probs.mean().item():.4f}", end = " - ")
    print("\n")