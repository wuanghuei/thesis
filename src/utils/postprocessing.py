from collections import defaultdict
import torch
from src.utils.helpers import calculate_temporal_iou

def resolve_cross_class_overlaps(merged_detections):
    """Giải quyết chồng lấp giữa các lớp sau khi merge, không cho phép bất kỳ frame nào bị chồng lấn"""
    for video_id, detections in merged_detections.items():
        # Sắp xếp theo confidence giảm dần
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Khởi tạo mảng detections mới không có chồng lấp
        final_detections = []
        
        # Tìm frame cuối cùng trong tất cả detections
        max_frame = max([det['end_frame'] for det in detections]) if detections else 0
        frames_occupied = [False] * (max_frame + 1)
        
        for det in detections:
            start = det['start_frame']
            end = det['end_frame']
            
            # Kiểm tra overlap - nếu bất kỳ frame nào đã bị chiếm, bỏ qua detection này
            overlap = False
            for t in range(start, end):
                if t < len(frames_occupied) and frames_occupied[t]:
                    overlap = True
                    break
            
            if not overlap:
                # Thêm detection và đánh dấu tất cả frame của nó là đã chiếm
                for t in range(start, end):
                    if t < len(frames_occupied):
                        frames_occupied[t] = True
                final_detections.append(det)
        
        # Cập nhật lại danh sách detections cho video này
        merged_detections[video_id] = final_detections
    
    return merged_detections

def merge_cross_window_detections(all_window_detections, all_window_metadata, iou_threshold=0.2, confidence_threshold=0.15):
    """
    Kết hợp detections từ các sliding window chồng lấp
    
    Args:
        all_window_detections: List các detections từ mỗi window [window_idx][detection_idx]
        all_window_metadata: Thông tin về mỗi window (video_id, start_idx, end_idx)
        iou_threshold: Ngưỡng IoU để kết hợp các detections liên tiếp
        confidence_threshold: Ngưỡng confidence để chấp nhận kết hợp
        
    Returns:
        merged_detections: Danh sách các detections đã kết hợp xuyên window
    """
    # Tổ chức các detections theo video_id và action_id
    video_detections = defaultdict(lambda: defaultdict(list))
    
    for window_idx, (window_dets, meta) in enumerate(zip(all_window_detections, all_window_metadata)):
        video_id = meta['video_id']
        start_idx = meta['start_idx']  # Vị trí bắt đầu của window trong video
        
        for det in window_dets:
            action_id = det['action_id']
            # Chuyển coordinates từ window-relative sang global video coordinates
            global_start = start_idx + det['start_frame']
            global_end = start_idx + det['end_frame']
            confidence = det['confidence']
            
            video_detections[video_id][action_id].append({
                'start_frame': global_start,
                'end_frame': global_end,
                'confidence': confidence,
                'window_idx': window_idx
            })
    
    # Kết hợp các detections thuộc cùng một action trong mỗi video
    merged_results = {}
    for video_id, action_dets in video_detections.items():
        merged_results[video_id] = []
        
        for action_id, dets in action_dets.items():
            # Sắp xếp theo vị trí bắt đầu
            dets = sorted(dets, key=lambda x: x['start_frame'])
            
            # Kết hợp các detections bị ngắt do window size
            i = 0
            while i < len(dets):
                current = dets[i]
                merged = dict(current)  # Copy để không thay đổi detection gốc
                
                j = i + 1
                while j < len(dets):
                    next_det = dets[j]
                    
                    # Kiểm tra xem hai detections có khả năng là một hành động bị cắt không
                    overlap = min(merged['end_frame'], next_det['end_frame']) - max(merged['start_frame'], next_det['start_frame'])
                    overlap_ratio = overlap / min(merged['end_frame'] - merged['start_frame'], next_det['end_frame'] - next_det['start_frame'])
                    
                    time_diff = abs(next_det['start_frame'] - merged['end_frame'])
                    
                    # Điều kiện để kết hợp: có overlap hoặc cách nhau không quá xa
                    if (overlap_ratio >= iou_threshold or time_diff <= 5) and \
                       (merged['confidence'] + next_det['confidence']) / 2 >= confidence_threshold:
                        # Mở rộng detection hiện tại
                        merged['start_frame'] = min(merged['start_frame'], next_det['start_frame'])
                        merged['end_frame'] = max(merged['end_frame'], next_det['end_frame'])
                        merged['confidence'] = (merged['confidence'] * (merged['end_frame'] - merged['start_frame']) + 
                                             next_det['confidence'] * (next_det['end_frame'] - next_det['start_frame'])) / \
                                             ((merged['end_frame'] - merged['start_frame']) + 
                                             (next_det['end_frame'] - next_det['start_frame']))
                        dets.pop(j)  # Loại bỏ detection đã kết hợp
                    else:
                        j += 1
                
                merged_results[video_id].append({
                    'action_id': action_id,
                    'start_frame': merged['start_frame'],
                    'end_frame': merged['end_frame'],
                    'confidence': merged['confidence']
                })
                
                i += 1
    
    return merged_results

def post_process(model, action_probs, start_probs, end_probs, class_thresholds, boundary_threshold, nms_threshold, min_segment_length):
    """Post-processing đơn giản hóa:
    1. Tìm start/end candidates > boundary_threshold.
    2. Tạo tất cả các cặp (start, end) hợp lệ (đúng min_length).
    3. Lọc các cặp dựa trên action_score trung bình > class_threshold.
    4. Tính confidence kết hợp.
    5. Áp dụng NMS cho cùng lớp trong cửa sổ.
    (Loại bỏ xử lý overlap khác lớp trong cửa sổ)
    """
    batch_size, seq_len, num_classes = action_probs.shape
    all_detections_batch = []
    
    for b in range(batch_size):
        detections_window = [] # Detections cho cửa sổ hiện tại
        
        for c in range(num_classes):
            action_score_c = action_probs[b, :, c]  # (T,)
            start_score_c = start_probs[b, :, c]    # (T,)
            end_score_c = end_probs[b, :, c]        # (T,)
            class_threshold_c = class_thresholds[c]

            # 1. Tìm start/end candidates
            start_indices = torch.where(start_score_c > boundary_threshold)[0]
            end_indices = torch.where(end_score_c > boundary_threshold)[0]

            # === ADD DEBUG FOR CLASS 2 ===
            if c == 2:
                print(f"DEBUG Class 2 (window {b}): Max Start Score = {start_score_c.max().item():.4f}, Max End Score = {end_score_c.max().item():.4f}")
                print(f"DEBUG Class 2 (window {b}): Num Start Indices (> {boundary_threshold}) = {len(start_indices)}, Num End Indices (> {boundary_threshold}) = {len(end_indices)}")
            # === END DEBUG ===

            if len(start_indices) == 0 or len(end_indices) == 0:
                continue
            
            proposals_class_c = []
            # 2. Tạo tất cả các cặp (start, end) hợp lệ
            for start_idx_tensor in start_indices:
                start_idx = start_idx_tensor.item()
                # Chỉ xét end_indices sau start_idx
                valid_end_indices = end_indices[end_indices > start_idx]

                for end_idx_tensor in valid_end_indices:
                    end_idx = end_idx_tensor.item()

                    # Kiểm tra độ dài tối thiểu
                    if (end_idx - start_idx) >= min_segment_length:
                        # 3. Lọc dựa trên action score trung bình
                        segment_action_score = action_score_c[start_idx:end_idx].mean().item()

                        if segment_action_score > class_threshold_c:
                            # 4. Tính confidence kết hợp
                            start_conf = start_score_c[start_idx].item()
                            # Lấy điểm end của frame cuối cùng TRONG segment (end_idx là exclusive)
                            # Đảm bảo end_idx-1 không nhỏ hơn start_idx
                            effective_end_idx = max(start_idx, end_idx - 1)
                            end_conf = end_score_c[effective_end_idx].item()

                            confidence = (segment_action_score + start_conf + end_conf) / 3.0

                            proposals_class_c.append({
                        'action_id': c,
                                'start_frame': start_idx,
                                'end_frame': end_idx, # end_idx là exclusive
                                'confidence': confidence
                            })

            # Thêm proposals của lớp này vào danh sách chung của window
            detections_window.extend(proposals_class_c)

        # 5. Áp dụng NMS cho TẤT CẢ detections trong window (chỉ loại bỏ cùng lớp)
        # Sắp xếp trước khi vào NMS để đảm bảo tính nhất quán
        detections_window = sorted(detections_window, key=lambda x: x['confidence'], reverse=True)
        detections_window_nms = nms(detections_window, nms_threshold) # nms chỉ xử lý overlap CÙNG LỚP

        # KHÔNG còn xử lý overlap khác lớp ở đây
        # KHÔNG còn gọi validate_detections ở đây

        all_detections_batch.append(detections_window_nms)

    return all_detections_batch

def nms(detections, threshold):
    """Non-maximum suppression for action detections (cho cùng một lớp)"""
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections with IoU > threshold
        detections = [
            d for d in detections if 
            calculate_temporal_iou(
                (current['start_frame'], current['end_frame']),
                (d['start_frame'], d['end_frame'])
            ) <= threshold
        ]
    
    return keep
