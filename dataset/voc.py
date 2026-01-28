import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


CLASSES = sorted(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'road_object'])
CLASSES = ['background'] + CLASSES


def parse_pascal_annotation(file_path):
    """Parse a Pascal VOC-style annotation text file."""
    with open(file_path, 'r') as f:
        content = f.read()

    filename = re.search(r'Image filename\s*:\s*"(.*?)"', content)
    if not filename:
        raise ValueError(f"Filename not found in {file_path}")
    filename = filename.group(1)

    size = re.search(r'Image size \(X x Y x C\)\s*:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)', content)
    if not size:
        raise ValueError(f"Image size not found in {file_path}")
    width, height, _ = map(int, size.groups())

    pattern = (
        r'Original label for object \d+ "(.*?)"\s*:\s*".*?"\n'
        r'Bounding box for object \d+ ".*?" \(Xmin, Ymin\) - \(Xmax, Ymax\) : '
        r'\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)'
    )
    objects = re.findall(pattern, content)
    annotations = [{"bbox": [int(x) for x in obj[1:]], "category_name": obj[0].lower()} for obj in objects]
    return filename, width, height, annotations


class SynchronizedAugmentation:
    """Apply identical augmentations to camera and radar images."""

    def __init__(self, is_train=True, num_augmentations=1, im_size=300):
        self.num_augmentations = num_augmentations
        if is_train:
            self.aug = iaa.Sequential([
                iaa.Resize({"height": im_size, "width": im_size}),
                iaa.Fliplr(0.5),
                iaa.Rotate((-10, 10)),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                           scale=(0.9, 1.1), shear=(-8, 8)),
                iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),
                iaa.LinearContrast((0.75, 1.25)),
                iaa.AddToBrightness((-30, 30)),
                iaa.GaussianBlur(sigma=(0, 0.5))
            ])
        else:
            self.aug = iaa.Sequential([iaa.Resize({"height": im_size, "width": im_size})])

    def _clip_boxes(self, bbs, shape):
        clipped = []
        for bb in bbs.bounding_boxes:
            x1, y1 = max(0, bb.x1), max(0, bb.y1)
            x2, y2 = min(shape[1] - 1, bb.x2), min(shape[0] - 1, bb.y2)
            if x2 > x1 and y2 > y1:
                clipped.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=bb.label))
        return BoundingBoxesOnImage(clipped, shape=shape)

    def __call__(self, cam_img, rad_img, bbs):
        results = []
        for _ in range(1 + self.num_augmentations):
            det = self.aug.to_deterministic()
            cam_aug = det(image=np.array(cam_img))
            rad_aug = det(image=np.array(rad_img))
            bbs_aug = self._clip_boxes(det(bounding_boxes=bbs), cam_aug.shape)
            results.append((Image.fromarray(cam_aug), Image.fromarray(rad_aug), bbs_aug))
        return results


class PascalDataset(Dataset):
    """Dataset for camera-radar fusion with Pascal VOC annotations."""

    def __init__(self, annotation_dir, cam_dir, radar_dir, transform=None,
                 is_train=True, im_size=300, num_augmentations=1):
        self.cam_dir = cam_dir
        self.radar_dir = radar_dir
        self.transform = transform or SynchronizedAugmentation(is_train, num_augmentations, im_size)
        self.num_augmentations = num_augmentations

        self.label2idx = {cls: i for i, cls in enumerate(CLASSES)}
        self.idx2label = {i: cls for cls, i in self.label2idx.items()}

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images_info = self._load_annotations(annotation_dir)
        print(f"Loaded {len(self.images_info)} images, {len(CLASSES)} classes")

    def _load_annotations(self, annotation_dir):
        infos = []
        for f in os.listdir(annotation_dir):
            if not f.endswith('.txt'):
                continue
            try:
                filename, w, h, anns = parse_pascal_annotation(os.path.join(annotation_dir, f))
            except ValueError:
                continue

            cam_path = os.path.join(self.cam_dir, filename)
            if not os.path.isfile(cam_path):
                continue

            rad_path = os.path.join(self.radar_dir, filename.replace("_cam.jpg", "_radar.png"))
            if not os.path.isfile(rad_path):
                rad_path = None

            dets = [{"bbox": a['bbox'], "label": self.label2idx.get(a['category_name'], 0)}
                    for a in anns]
            infos.append({"filename": cam_path, "radar_filename": rad_path,
                          "width": w, "height": h, "detections": dets})
        return infos

    def __len__(self):
        return len(self.images_info) * (1 + self.num_augmentations)

    def __getitem__(self, index):
        try:
            sample_idx = index // (1 + self.num_augmentations)
            aug_idx = index % (1 + self.num_augmentations)
            info = self.images_info[sample_idx]

            cam_img = Image.open(info['filename']).convert("RGB")
            if info['radar_filename']:
                rad_img = Image.open(info['radar_filename']).convert("RGB")
            else:
                rad_img = Image.new("RGB", cam_img.size, (0, 0, 0))

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=d['bbox'][0], y1=d['bbox'][1], x2=d['bbox'][2], y2=d['bbox'][3],
                            label=self.idx2label.get(d['label'], 'background'))
                for d in info['detections']
            ], shape=(cam_img.height, cam_img.width, 3))

            aug_results = self.transform(cam_img, rad_img, bbs)
            cam_aug, rad_aug, bbs_aug = aug_results[aug_idx]

            if len(bbs_aug.bounding_boxes) > 0:
                boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in bbs_aug.bounding_boxes], dtype=torch.float32)
                labels = torch.tensor([self.label2idx.get(b.label.lower(), 0) for b in bbs_aug.bounding_boxes], dtype=torch.int64)
                boxes[:, [0, 2]] /= cam_aug.width
                boxes[:, [1, 3]] /= cam_aug.height
                boxes = boxes.clamp(0, 1)
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, labels = boxes[valid], labels[valid]
                difficult = torch.zeros(len(labels), dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                difficult = torch.zeros((0,), dtype=torch.int64)

            target = {"bboxes": boxes, "labels": labels, "difficult": difficult}
            cam_t = self.normalize(T.ToTensor()(cam_aug))
            rad_t = self.normalize(T.ToTensor()(rad_aug))
            return cam_t, rad_t, target, info['filename']
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return None


def collate_fn(batch):
    """Handle variable-length targets and filter failed samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    cam, rad, targets, paths = zip(*batch)
    return torch.stack(cam), torch.stack(rad), list(targets), list(paths)
