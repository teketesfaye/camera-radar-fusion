import torch
import argparse
import os
import sys
import yaml
import random
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ssd import SSD
from dataset.voc import PascalDataset, SynchronizedAugmentation, collate_fn


def get_iou(boxA, boxB):
    """IoU between two boxes in x1y1x2y2 format."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return inter / float(areaA + areaB - inter)


def compute_ap(cls_dets, gt_boxes, gt_matched, num_gts, num_difficults, method='area'):
    """Compute average precision for one class."""
    tp = [0] * len(cls_dets)
    fp = [0] * len(cls_dets)

    for det_idx, (im_idx, det_pred, im_gt_difficults) in enumerate(cls_dets):
        im_gts = gt_boxes[im_idx]
        best_iou, best_gt = -1, -1
        for gt_idx, gt_box in enumerate(im_gts):
            iou = get_iou(det_pred[:-1], gt_box)
            if iou > best_iou:
                best_iou, best_gt = iou, gt_idx

        if best_iou >= 0.5 and not im_gt_difficults[best_gt]:
            if not gt_matched[im_idx][best_gt]:
                gt_matched[im_idx][best_gt] = True
                tp[det_idx] = 1
            else:
                fp[det_idx] = 1
        else:
            fp[det_idx] = 1

    tp, fp = np.cumsum(tp), np.cumsum(fp)
    eps = np.finfo(np.float32).eps
    recalls = tp / max(num_gts - num_difficults, eps)
    precisions = tp / np.maximum(tp + fp, eps)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    i = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])


def compute_metrics(detections, ground_truths, difficults):
    """Compute mAP and wmAP across all classes."""
    gt_labels = {k for gt in ground_truths for k in gt if k != 'background'}
    gt_labels = sorted(gt_labels)

    all_aps, counts = {}, {}
    for label in gt_labels:
        cls_dets = []
        for im_idx, im_dets in enumerate(detections):
            if label in im_dets:
                for d in im_dets[label]:
                    cls_dets.append((im_idx, d, difficults[im_idx].get(label, [])))
        cls_dets.sort(key=lambda x: -x[1][-1])

        gt_matched = [[False] * len(gt.get(label, [])) for gt in ground_truths]
        gt_per_img = [gt.get(label, []) for gt in ground_truths]
        num_gts = sum(len(g) for g in gt_per_img)
        num_diff = sum(sum(d.get(label, [])) for d in difficults)
        counts[label] = num_gts

        # Repack for compute_ap
        packed = [(im_idx, det, difficults[im_idx].get(label, [0] * len(gt_per_img[im_idx])))
                  for im_idx, det, _ in cls_dets]
        if num_gts > 0:
            all_aps[label] = compute_ap(packed, gt_per_img, gt_matched, num_gts, num_diff)
        else:
            all_aps[label] = np.nan

    valid = {k: v for k, v in all_aps.items() if not np.isnan(v)}
    mAP = np.mean(list(valid.values())) if valid else 0.0
    total = sum(counts[k] for k in valid)
    wmAP = sum(counts[k] * valid[k] for k in valid) / total if total > 0 else 0.0
    return mAP, wmAP, all_aps


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data_params', {})
    model_config = config['model_params']
    train_config = config.get('train_params', {})

    # Dataset
    val_dataset = PascalDataset(
        annotation_dir=data_config['val_annotation_dir'],
        cam_dir=data_config['val_cam_dir'],
        radar_dir=data_config['val_radar_dir'],
        transform=SynchronizedAugmentation(is_train=False, im_size=300),
        is_train=False, im_size=300, num_augmentations=0
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # Load model
    model = SSD(config=model_config, num_classes=8)
    ckpt = os.path.join(train_config.get('task_name', ''), train_config.get('ckpt_name', ''))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # Measure inference time
    if args.infer_time:
        dummy_cam = torch.randn(1, 3, 300, 300).to(device)
        dummy_rad = torch.randn(1, 3, 300, 300).to(device)
        with torch.no_grad():
            model(dummy_cam, dummy_rad)  # warmup
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            model(dummy_cam, dummy_rad)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print(f"Fusion inference: {(time.time()-t0)*1000:.1f} ms")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            model(dummy_cam)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print(f"Camera-only inference: {(time.time()-t0)*1000:.1f} ms")

    # Evaluate
    detections, ground_truths, difficults = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None:
                continue
            cam, rad, targets, _ = batch
            cam, rad = cam.to(device), rad.to(device)
            for t in targets:
                t['bboxes'] = t['bboxes'].float().to(device)
                t['labels'] = t['labels'].long().to(device)

            _, preds = model(cam, rad)

            for i, pred in enumerate(preds):
                det_dict = defaultdict(list)
                for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                    name = val_dataset.idx2label.get(label.item(), 'background')
                    if name != 'background':
                        det_dict[name].append(box.cpu().numpy().tolist() + [score.item()])
                detections.append(dict(det_dict))

                gt_dict, diff_dict = defaultdict(list), defaultdict(list)
                for box, label, diff in zip(targets[i]['bboxes'], targets[i]['labels'], targets[i]['difficult']):
                    name = val_dataset.idx2label.get(label.item(), 'background')
                    if name != 'background':
                        gt_dict[name].append(box.cpu().numpy().tolist())
                        diff_dict[name].append(diff.item())
                ground_truths.append(dict(gt_dict))
                difficults.append(dict(diff_dict))

    mAP, wmAP, class_aps = compute_metrics(detections, ground_truths, difficults)
    print("\nResults:")
    for cls, ap in class_aps.items():
        print(f"  {cls}: {ap:.4f}")
    print(f"  mAP:  {mAP:.4f}")
    print(f"  wmAP: {wmAP:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', required=True)
    parser.add_argument('--infer_time', action='store_true')
    infer(parser.parse_args())
