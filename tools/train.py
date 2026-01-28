import torch
import argparse
import os
import sys
import numpy as np
import yaml
import random
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ssd import SSD
from dataset.voc import PascalDataset, SynchronizedAugmentation, collate_fn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


def train(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model_params']
    train_config = config['train_params']
    data_config = config.get('data_params', {})

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Reproducibility
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset
    train_dataset = PascalDataset(
        annotation_dir=data_config['train_annotation_dir'],
        cam_dir=data_config['train_cam_dir'],
        radar_dir=data_config['train_radar_dir'],
        transform=SynchronizedAugmentation(is_train=True, num_augmentations=2, im_size=300),
        is_train=True, im_size=300, num_augmentations=2
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config['batch_size'], shuffle=True,
        num_workers=train_config.get('num_workers', 4), collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )

    # Model
    model = SSD(config=model_config, num_classes=8).to(device)

    # Resume from checkpoint if exists
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Resumed from {ckpt_path}')

    os.makedirs(train_config['task_name'], exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=train_config['lr'], weight_decay=5e-4, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    acc_steps = train_config.get('acc_steps', 1)

    # Training loop
    for epoch in range(train_config['num_epochs']):
        model.train()
        cls_losses, loc_losses = [], []

        for idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch is None:
                continue

            cam, rad, targets, _ = batch
            cam = cam.to(device, non_blocking=True)
            rad = rad.to(device, non_blocking=True)
            for t in targets:
                t['boxes'] = t['bboxes'].float().to(device)
                t['labels'] = t['labels'].long().to(device)

            losses, _ = model(cam, rad, targets)
            loss = (losses['classification'] + losses['bbox_regression']) / acc_steps
            loss.backward()

            cls_losses.append(losses['classification'].item())
            loc_losses.append(losses['bbox_regression'].item())

            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if torch.isnan(loss):
                print('NaN loss detected, stopping.')
                return

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        print(f"Epoch {epoch+1}: cls={np.mean(cls_losses):.4f} loc={np.mean(loc_losses):.4f}")
        torch.save(model.state_dict(), ckpt_path)

    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/voc.yaml')
    train(parser.parse_args())
