import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import json
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from tqdm.autonotebook import tqdm


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class KeypointsDataset(Dataset):
    def __init__(self, data_path, anotaion_path, transforms = None):
        self.data_paths = data_path
        with open(anotaion_path, "r") as f:
            self.data = json.load(f)
        self.transforms = transforms


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_paths, f"{item['id']}.png")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if self.transforms:
            img = self.transforms(img)

        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)

        kps[::2] *= 224.0 / w  # Adjust x coordinates
        kps[1::2] *= 224.0 / h  # Adjust y coordinates
        return img, kps

def get_args():
    parser = argparse.ArgumentParser(description="Train tennis court keypoints")
    parser.add_argument("--num_epochs", "-n", type=int, default=80)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", "-l", type=float, default=0.0001)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-p", type=str, default=None)
    parser.add_argument("--tensorboard_dir", "-t", type=str, default="tensorboard")

    args = parser.parse_args()
    return args


def train(args):

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if os.path.exists(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
    os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)

    transforms = Compose([
        ToTensor(),
        Resize((224, 224), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = KeypointsDataset("../data/images","../data/data_train.json", transforms)
    val_dataset = KeypointsDataset("../data/images","../data/data_val.json", transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 14*2)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    lr = args.lr
    opimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    num_iter = len(train_loader)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_MSE = checkpoint["best_MSE"]
        model.load_state_dict(checkpoint["model_params"])
        opimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_MSE = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        # train
        model.train()
        progressbar = tqdm(train_loader, colour="GREEN")
        all_losses = []
        for iter, (images, kps) in enumerate(progressbar):

            if epoch < args.num_epochs / 2:
                new_lr = args.lr
            elif args.num_epochs / 2 < epoch < 4 * args.num_epochs / 5:
                new_lr = args.lr / 10
            else:
                new_lr = args.lr / 100
            if new_lr != lr:
                lr = new_lr
                for param_groups in opimizer.param_groups:
                    param_groups['lr'] = lr

            images = images.to(device)
            kps = kps.to(device)

            output = model(images)
            loss = criterion(output, kps)
            all_losses.append(loss.item())

            opimizer.zero_grad()
            loss.backward()
            opimizer.step()

            loss_value = np.mean(all_losses)
            progressbar.set_description(f"epoch: {epoch + 1}/{args.num_epochs}. Loss: {loss_value}")
            writer.add_scalar('Train/Loss', loss_value, epoch * num_iter + iter)

        # validation

        all_losses = []
        model.eval()
        progressbar = tqdm(val_loader, colour = "BLUE")
        with torch.no_grad():
            for iter, (images, kps) in enumerate(progressbar):
                images = images.to(device)
                kps = kps.to(device)

                predict = model(images)
                loss = criterion(predict, kps)
                all_losses.append(loss.item())

        loss_value = np.mean(all_losses)
        writer.add_scalar('Val/Loss', loss_value, epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "best_MSE":loss_value,
            "model_params": model.state_dict(),
            "optimizer": opimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if best_MSE > loss_value:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            best_MSE = loss_value
    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)