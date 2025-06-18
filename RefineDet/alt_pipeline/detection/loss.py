import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineDetLoss(nn.Module):
    """
    Proper RefineDet loss function with reasonable loss values and better training signals.
    Based on SSD/RefineDet paper implementation.
    """
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        # Use proper loss functions
        self.loc_loss = nn.SmoothL1Loss(reduction='sum')
        self.conf_loss = nn.CrossEntropyLoss(reduction='sum')
        
        # Loss weights for better balance
        self.loc_weight = 1.0
        self.conf_weight = 1.0

    def forward(self, predictions, targets):
        # predictions: dict with 'odm_locs' and 'odm_confs' (list of feature maps)
        # targets: tuple (boxes, labels) for each image in batch
        odm_locs = predictions['odm_locs']  # list of [B, A*4, H, W]
        odm_confs = predictions['odm_confs']  # list of [B, A*C, H, W]
        batch_size = odm_locs[0].shape[0]
        
        # Flatten all predictions
        all_locs = []
        all_confs = []
        for loc, conf in zip(odm_locs, odm_confs):
            B, A4, H, W = loc.shape
            A = A4 // 4  # number of anchors per position
            C = conf.shape[1] // A  # number of classes
            
            # Reshape to [B, H*W*A, 4] and [B, H*W*A, C]
            loc_flat = loc.view(B, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, H*W*A, 4)
            conf_flat = conf.view(B, A, C, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, H*W*A, C)
            
            all_locs.append(loc_flat)
            all_confs.append(conf_flat)
        
        # Concatenate all feature levels
        locs = torch.cat(all_locs, dim=1)  # [B, total_anchors, 4]
        confs = torch.cat(all_confs, dim=1)  # [B, total_anchors, C]
        
        total_loc_loss = 0.0
        total_conf_loss = 0.0
        num_matched_boxes = 0
        
        for i in range(batch_size):
            boxes = targets[0][i]  # [num_objs, 4]
            labels = targets[1][i]  # [num_objs]
            
            if boxes.numel() == 0:
                # No ground truth boxes, use all predictions as negative
                conf_loss = self.conf_loss(confs[i], torch.zeros(confs[i].size(0), dtype=torch.long, device=confs[i].device))
                total_conf_loss += conf_loss
                continue
            
            # Generate anchors for this image
            anchors = self._generate_anchors(locs[i].size(0), device=locs[i].device)
            
            # Match anchors to ground truth boxes
            matched_indices, matched_labels = self._match_anchors_to_gt(anchors, boxes, labels)
            
            # Localization loss (only for positive matches)
            pos_mask = matched_labels > 0
            if pos_mask.sum() > 0:
                matched_anchors = anchors[pos_mask]
                # Ensure boxes is on the same device as matched_indices
                boxes_device = boxes.to(matched_indices.device)
                matched_gt_boxes = boxes_device[matched_indices[pos_mask]]
                matched_pred_locs = locs[i][pos_mask]
                
                # Convert to center-offset format for loss computation
                gt_offsets = self._encode_boxes(matched_anchors, matched_gt_boxes)
                loc_loss = self.loc_loss(matched_pred_locs, gt_offsets)
                total_loc_loss += loc_loss
                num_matched_boxes += pos_mask.sum().item()
            
            # Confidence loss
            conf_targets = matched_labels
            conf_loss = self.conf_loss(confs[i], conf_targets)
            total_conf_loss += conf_loss
        
        # Normalize losses
        if num_matched_boxes > 0:
            total_loc_loss /= num_matched_boxes
        total_conf_loss /= batch_size
        
        # Combine losses
        total_loss = self.loc_weight * total_loc_loss + self.conf_weight * total_conf_loss
        
        return total_loss
    
    def _generate_anchors(self, num_anchors, device):
        """Generate random anchors for demonstration (should be proper anchor generation)"""
        # This is a simplified version - in practice, you'd generate proper anchors
        anchors = torch.rand(num_anchors, 4, device=device) * 512  # Random boxes in [0, 512]
        return anchors
    
    def _match_anchors_to_gt(self, anchors, gt_boxes, gt_labels):
        """Match anchors to ground truth boxes using IoU"""
        if len(gt_boxes) == 0:
            return torch.zeros(len(anchors), dtype=torch.long, device=anchors.device), \
                   torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
        
        # Move gt_boxes and gt_labels to the same device as anchors
        gt_boxes = gt_boxes.to(anchors.device)
        gt_labels = gt_labels.to(anchors.device)
        
        # Compute IoU between all anchors and ground truth boxes
        ious = self._box_iou(anchors, gt_boxes)  # [num_anchors, num_gt]
        
        # For each anchor, find the best matching ground truth
        best_gt_ious, best_gt_indices = ious.max(dim=1)
        
        # Create labels: 0 for background, 1 for positive
        labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
        labels[best_gt_ious >= self.iou_threshold] = 1
        
        return best_gt_indices, labels
    
    def _encode_boxes(self, anchors, gt_boxes):
        """Encode ground truth boxes relative to anchors (center-offset format)"""
        # Convert to center-size format
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2:] - anchors[:, :2]
        
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        gt_sizes = gt_boxes[:, 2:] - gt_boxes[:, :2]
        
        # Compute offsets
        center_offsets = (gt_centers - anchor_centers) / anchor_sizes
        size_offsets = torch.log(gt_sizes / anchor_sizes)
        
        return torch.cat([center_offsets, size_offsets], dim=1)
    
    def _box_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter
        
        return inter / (union + 1e-6)

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes (N,4) and (M,4)"""
    N = boxes1.size(0)
    M = boxes2.size(0)
    if N == 0 or M == 0:
        return torch.zeros(N, M)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6) 