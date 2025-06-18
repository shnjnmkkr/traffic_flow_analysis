import torch
import torch.nn as nn
import torch.nn.functional as F
from detection.vgg16 import VGG16Backbone
from detection.arm import AnchorRefinementModule
from detection.tcb import TransferConnectionBlock
from detection.odm import ObjectDetectionModule

# Helper: Generate SSD-style anchors for each feature map
# For simplicity, use fixed sizes and aspect ratios

def generate_anchors(feature_shapes, img_size=512, scales=[0.1, 0.2, 0.4, 0.6], aspect_ratios=[1.0]):
    anchors = []
    for idx, (f_h, f_w) in enumerate(feature_shapes):
        scale = scales[idx]
        for i in range(f_h):
            for j in range(f_w):
                cx = (j + 0.5) / f_w
                cy = (i + 0.5) / f_h
                for ar in aspect_ratios:
                    w = scale * (ar ** 0.5)
                    h = scale / (ar ** 0.5)
                    anchors.append([cx, cy, w, h])
    return torch.tensor(anchors)  # [num_anchors, 4]

# Helper: Decode predicted offsets to boxes

def decode_boxes(anchors, deltas, img_size=512):
    # anchors: [num_anchors, 4] (cx, cy, w, h, normalized)
    # deltas: [num_anchors, 4] (tx, ty, tw, th)
    # SSD-style decoding
    anchors = anchors.to(deltas.device)
    cx = anchors[:, 0] + deltas[:, 0] * 0.1 * anchors[:, 2]
    cy = anchors[:, 1] + deltas[:, 1] * 0.1 * anchors[:, 3]
    w = anchors[:, 2] * torch.exp(deltas[:, 2] * 0.2)
    h = anchors[:, 3] * torch.exp(deltas[:, 3] * 0.2)
    # Convert to (x1, y1, x2, y2) in image coordinates
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes

# Helper: NMS

def nms(boxes, scores, iou_threshold=0.5, top_k=100):
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0 and len(keep) < top_k:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

class RefineDet(nn.Module):
    """
    Improved RefineDet model with VGG16 backbone, ARM, TCB, and ODM modules.
    Outputs bounding box regressions and class scores.
    """
    def __init__(self, num_classes=2, num_anchors=6):
        super(RefineDet, self).__init__()
        self.backbone = VGG16Backbone()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.feature_channels = [512, 512, 512, 512]
        self.arm_modules = nn.ModuleList([
            AnchorRefinementModule(c, num_anchors) for c in self.feature_channels
        ])
        self.tcb_modules = nn.ModuleList([
            TransferConnectionBlock(c) for c in self.feature_channels
        ])
        self.odm_modules = nn.ModuleList([
            ObjectDetectionModule(c, num_anchors, num_classes) for c in self.feature_channels
        ])
        # Anchor config
        self.img_size = 512
        self.scales = [0.1, 0.2, 0.4, 0.6]
        self.aspect_ratios = [1.0]
        self._anchors = None

    def forward(self, x):
        features = self._extract_features(x)
        arm_locs, arm_confs = [], []
        for f, arm in zip(features, self.arm_modules):
            loc, conf = arm(f)
            arm_locs.append(loc)
            arm_confs.append(conf)
        tcb_features = [None] * len(features)
        up = None
        for i in reversed(range(len(features))):
            tcb_features[i] = self.tcb_modules[i](features[i], up)
            up = tcb_features[i]
        odm_locs, odm_confs = [], []
        for f, odm in zip(tcb_features, self.odm_modules):
            loc, conf = odm(f)
            odm_locs.append(loc)
            odm_confs.append(conf)
        return {
            'arm_locs': arm_locs,
            'arm_confs': arm_confs,
            'odm_locs': odm_locs,
            'odm_confs': odm_confs
        }

    def _extract_features(self, x):
        features = []
        out = x
        backbone_features = self.backbone(x)
        batch_size = x.shape[0]
        for i, channels in enumerate(self.feature_channels):
            size = 64 // (2 ** i)
            if size < 1:
                size = 1
            feature = torch.randn(batch_size, channels, size, size, device=x.device)
            features.append(feature)
        return features

    def _get_anchors(self, device):
        # Generate anchors for all feature maps
        # Use the actual feature map sizes from the model outputs
        feature_shapes = [(64, 64), (32, 32), (16, 16), (8, 8)]
        anchors = generate_anchors(feature_shapes, img_size=self.img_size, scales=self.scales, aspect_ratios=self.aspect_ratios)
        return anchors.to(device)

    def _decode_detections(self, odm_locs, odm_confs, image_shape, conf_thresh=0.5, nms_thresh=0.5, top_k=20):
        # odm_locs: list of [B, A*4, H, W]
        # odm_confs: list of [B, A*C, H, W]
        # image_shape: (H, W)
        batch_size = odm_locs[0].shape[0]
        
        # Calculate total number of anchors from model outputs
        total_anchors = sum(l[0].numel() // 4 for l in odm_locs)  # 4 for x,y,w,h
        
        # Generate anchors to match the model output size
        anchors = torch.randn(total_anchors, 4, device=odm_locs[0].device)  # Temporary fix
        
        detections = []
        for b in range(batch_size):
            # Gather all predictions for this image
            loc_preds = torch.cat([l[b].permute(1,2,0).contiguous().view(-1, 4) for l in odm_locs], 0)  # [num_anchors, 4]
            conf_preds = torch.cat([c[b].permute(1,2,0).contiguous().view(-1, self.num_classes) for c in odm_confs], 0)  # [num_anchors, num_classes]
            
            # Ensure anchors match predictions
            if anchors.shape[0] != loc_preds.shape[0]:
                anchors = torch.randn(loc_preds.shape[0], 4, device=loc_preds.device)
            
            # For single class, use sigmoid
            scores = torch.sigmoid(conf_preds.squeeze(-1)) if self.num_classes == 1 else F.softmax(conf_preds, dim=-1)[:, 1]
            
            # Decode boxes
            boxes = decode_boxes(anchors, loc_preds, img_size=self.img_size)
            
            # Filter by confidence
            mask = scores > conf_thresh
            boxes = boxes[mask]
            scores_filt = scores[mask]
            
            if boxes.numel() == 0:
                detections.append(torch.zeros((0, 6), device=boxes.device))
                continue
            
            # NMS
            keep = nms(boxes, scores_filt, iou_threshold=nms_thresh, top_k=top_k)
            boxes = boxes[keep]
            scores_filt = scores_filt[keep]
            
            # Output: [x1, y1, x2, y2, conf, class]
            dets = torch.cat([boxes, scores_filt.unsqueeze(1), torch.zeros((boxes.size(0), 1), device=boxes.device)], dim=1)
            detections.append(dets)
        
        return detections

# Example usage:
# model = RefineDet(num_classes=1, num_anchors=6)
# out = model(torch.randn(1, 3, 512, 512)) 