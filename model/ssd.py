import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict
from model.backbone import ResNetSSDBackbone


def get_iou(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes (N,4) and (M,4)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection
    return intersection / union


def boxes_to_targets(gt_boxes, default_boxes, weights=(10., 10., 5., 5.)):
    """Encode ground truth boxes as regression targets relative to default boxes."""
    w = default_boxes[:, 2] - default_boxes[:, 0]
    h = default_boxes[:, 3] - default_boxes[:, 1]
    cx = default_boxes[:, 0] + 0.5 * w
    cy = default_boxes[:, 1] + 0.5 * h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h

    return torch.stack([
        weights[0] * (gt_cx - cx) / w,
        weights[1] * (gt_cy - cy) / h,
        weights[2] * torch.log(gt_w / w),
        weights[3] * torch.log(gt_h / h),
    ], dim=1)


def apply_deltas(deltas, default_boxes, weights=(10., 10., 5., 5.)):
    """Decode predicted deltas back to box coordinates."""
    w = default_boxes[:, 2] - default_boxes[:, 0]
    h = default_boxes[:, 3] - default_boxes[:, 1]
    cx = default_boxes[:, 0] + 0.5 * w
    cy = default_boxes[:, 1] + 0.5 * h

    pred_cx = deltas[..., 0] / weights[0] * w + cx
    pred_cy = deltas[..., 1] / weights[1] * h + cy
    pred_w = torch.exp(deltas[..., 2] / weights[2]) * w
    pred_h = torch.exp(deltas[..., 3] / weights[3]) * h

    return torch.stack([
        pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h,
        pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h,
    ], dim=-1)


def generate_default_boxes(features, aspect_ratios, scales):
    """Generate SSD default boxes for all feature map scales."""
    default_boxes = []
    for k in range(len(features)):
        s_prime = math.sqrt(scales[k] * scales[k + 1])
        wh_pairs = [[s_prime, s_prime]]
        for ar in aspect_ratios[k]:
            sq = math.sqrt(ar)
            wh_pairs.append([scales[k] * sq, scales[k] / sq])

        feat_h, feat_w = features[k].shape[-2:]
        sx = ((torch.arange(0, feat_w) + 0.5) / feat_w).float()
        sy = ((torch.arange(0, feat_h) + 0.5) / feat_h).float()
        shift_y, shift_x = torch.meshgrid(sy, sx, indexing="ij")

        shifts = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)) * len(wh_pairs), dim=-1).reshape(-1, 2)
        wh = torch.as_tensor(wh_pairs).repeat(feat_h * feat_w, 1)
        default_boxes.append(torch.cat([shifts, wh], dim=1))

    all_boxes = torch.cat(default_boxes, dim=0)

    # Convert cx,cy,w,h to x1,y1,x2,y2 and replicate per batch
    dboxes = []
    for _ in range(features[0].size(0)):
        xyxy = torch.cat([all_boxes[:, :2] - 0.5 * all_boxes[:, 2:],
                          all_boxes[:, :2] + 0.5 * all_boxes[:, 2:]], dim=-1)
        dboxes.append(xyxy.to(features[0].device))
    return dboxes


class BAM(nn.Module):
    """Bottleneck Attention Module -- channel + spatial attention."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )
        self.integrate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Channel attention
        x_ca = self.channel_attn(x) * x
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        x_sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1)) * x_ca
        return self.integrate(x_sa)


class MultiModalFusion(nn.Module):
    """Fuse camera and radar features at each scale using concat + 1x1 conv + BAM."""

    def __init__(self, feature_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.reduce = nn.ModuleList([
            nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False) for ch in feature_channels
        ])
        self.attn = nn.ModuleList([
            BAM(ch, reduction=reduction, kernel_size=kernel_size) for ch in feature_channels
        ])

    def forward(self, cam_feats, rad_feats):
        fused = []
        for i, (cam, rad) in enumerate(zip(cam_feats, rad_feats)):
            concat = torch.cat([cam, rad], dim=1)
            reduced = self.reduce[i](concat)
            fused.append(self.attn[i](reduced))
        return fused


class SSD(nn.Module):
    """SSD object detector with camera-radar fusion via BAM attention."""

    def __init__(self, config: Dict, num_classes: int = 8):
        super().__init__()
        self.aspect_ratios = config['aspect_ratios']
        self.scales = config['scales'] + [1.0]
        self.num_classes = num_classes
        self.iou_threshold = config['iou_threshold']
        self.low_score_threshold = config['low_score_threshold']
        self.neg_pos_ratio = config['neg_pos_ratio']
        self.pre_nms_topK = config['pre_nms_topK']
        self.nms_threshold = config['nms_threshold']
        self.detections_per_img = config['detections_per_img']

        # Dual backbones for camera and radar
        self.features_camera = ResNetSSDBackbone(backbone='resnet18', pretrained=True)
        self.features_radar = ResNetSSDBackbone(backbone='resnet18', pretrained=False)

        # Fusion module
        out_channels = [512, 1024, 512, 256, 256, 256]
        self.fusion = MultiModalFusion(out_channels)

        # Per-scale prediction heads
        self.cls_heads = nn.ModuleList()
        self.bbox_reg_heads = nn.ModuleList()
        for ch, ar in zip(out_channels, self.aspect_ratios):
            n_boxes = len(ar) + 1  # +1 for the extra scale
            self.cls_heads.append(nn.Conv2d(ch, num_classes * n_boxes, 3, padding=1))
            self.bbox_reg_heads.append(nn.Conv2d(ch, 4 * n_boxes, 3, padding=1))
        self._init_heads()

    def _init_heads(self):
        for m in list(self.cls_heads) + list(self.bbox_reg_heads):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def compute_loss(self, targets, cls_logits, bbox_reg, default_boxes, matched_idxs):
        num_fg = 0
        bbox_loss = []
        cls_targets = []

        for tgt, reg, cls, dbox, midx in zip(targets, bbox_reg, cls_logits, default_boxes, matched_idxs):
            fg = torch.where(midx >= 0)[0]
            fg_match = midx[fg]
            num_fg += fg_match.numel()

            target_reg = boxes_to_targets(tgt["boxes"][fg_match], dbox[fg])
            bbox_loss.append(F.smooth_l1_loss(reg[fg], target_reg, reduction='sum'))

            gt_cls = torch.zeros(cls.size(0), dtype=tgt["labels"].dtype, device=tgt["labels"].device)
            gt_cls[fg] = tgt["labels"][fg_match]
            cls_targets.append(gt_cls)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Hard negative mining
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, self.num_classes), cls_targets.view(-1), reduction="none"
        ).view(cls_targets.size())

        fg_mask = cls_targets > 0
        num_neg = self.neg_pos_ratio * fg_mask.sum(1, keepdim=True)
        neg_loss = cls_loss.clone()
        neg_loss[fg_mask] = -float("inf")
        _, idx = neg_loss.sort(1, descending=True)
        bg_mask = idx.sort(1)[1] < num_neg

        N = max(1, num_fg)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[fg_mask].sum() + cls_loss[bg_mask].sum()) / N,
        }

    def forward(self, camera_img, radar_img=None, targets=None):
        cam_feats = self.features_camera(camera_img)

        # Camera-only or camera+radar fusion
        if radar_img is None:
            features = cam_feats
        else:
            rad_feats = self.features_radar(radar_img)
            features = self.fusion(cam_feats, rad_feats)

        # Collect predictions from all scales
        cls_logits, bbox_deltas = [], []
        for i, feat in enumerate(features):
            cls = self.cls_heads[i](feat)
            reg = self.bbox_reg_heads[i](feat)
            N, _, H, W = cls.shape
            cls = cls.view(N, -1, self.num_classes, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, self.num_classes)
            reg = reg.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
            cls_logits.append(cls)
            bbox_deltas.append(reg)

        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_deltas = torch.cat(bbox_deltas, dim=1)
        default_boxes = generate_default_boxes(features, self.aspect_ratios, self.scales)

        losses = {}
        detections = []

        if self.training:
            # Match default boxes to ground truth
            matched_idxs = []
            for dbox, tgt in zip(default_boxes, targets):
                if tgt["boxes"].numel() == 0:
                    matched_idxs.append(torch.full((dbox.size(0),), -1, dtype=torch.int64, device=dbox.device))
                    continue
                iou = get_iou(tgt["boxes"], dbox)
                vals, matches = iou.max(dim=0)
                matches[vals < self.iou_threshold] = -1
                # Ensure each GT has at least one match
                _, best = iou.max(dim=1)
                matches[best] = torch.arange(best.size(0), dtype=torch.int64, device=best.device)
                matched_idxs.append(matches)
            losses = self.compute_loss(targets, cls_logits, bbox_deltas, default_boxes, matched_idxs)
        else:
            # Inference: decode and NMS
            scores = F.softmax(cls_logits, dim=-1)
            for deltas_i, scores_i, dbox_i in zip(bbox_deltas, scores, default_boxes):
                boxes = apply_deltas(deltas_i, dbox_i).clamp_(0., 1.)
                pred_boxes, pred_scores, pred_labels = [], [], []

                for label in range(1, self.num_classes):
                    sc = scores_i[:, label]
                    keep = sc > self.low_score_threshold
                    sc, bx = sc[keep], boxes[keep]
                    if sc.numel() == 0:
                        continue
                    top_k = min(self.pre_nms_topK, sc.size(0))
                    sc, top_idx = sc.topk(top_k)
                    pred_boxes.append(bx[top_idx])
                    pred_scores.append(sc)
                    pred_labels.append(torch.full_like(sc, label, dtype=torch.int64))

                if pred_boxes:
                    pred_boxes = torch.cat(pred_boxes)
                    pred_scores = torch.cat(pred_scores)
                    pred_labels = torch.cat(pred_labels)

                    # Per-class NMS
                    keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                    for cid in torch.unique(pred_labels):
                        ci = torch.where(pred_labels == cid)[0]
                        nms_keep = torch.ops.torchvision.nms(pred_boxes[ci], pred_scores[ci], self.nms_threshold)
                        keep_mask[ci[nms_keep]] = True

                    keep = torch.where(keep_mask)[0]
                    if keep.numel() > self.detections_per_img:
                        _, top = pred_scores[keep].topk(self.detections_per_img)
                        keep = keep[top]

                    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
                else:
                    pred_boxes = torch.empty((0, 4), device=scores_i.device)
                    pred_scores = torch.empty((0,), device=scores_i.device)
                    pred_labels = torch.empty((0,), dtype=torch.int64, device=scores_i.device)

                detections.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})

        return losses, detections
