from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import urllib.request
import zipfile
from datetime import datetime

# Set torchaudio backend for macOS
torchaudio.set_audio_backend("soundfile")


def download_esc50_data(data_dir="./esc50_data"):
    """Download and extract ESC-50 dataset"""
    data_path = Path(data_dir)

    if data_path.exists() and (data_path / "audio").exists():
        print(f"ESC-50 data already exists at {data_path}")
        return data_path

    print("Downloading ESC-50 dataset...")
    data_path.mkdir(exist_ok=True)

    # Download the dataset
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = data_path / "esc50.zip"

    urllib.request.urlretrieve(url, zip_path)
    print("Download completed. Extracting...")

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Move contents from ESC-50-master to the main directory
    extracted_dir = data_path / "ESC-50-master"
    for item in extracted_dir.iterdir():
        item.rename(data_path / item.name)

    # Clean up
    extracted_dir.rmdir()
    zip_path.unlink()

    print(f"ESC-50 dataset extracted to {data_path}")
    return data_path


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]

        waveform, sample_rate = torchaudio.load(audio_path)
        # waveforms - [channels, samples]  - for 2 channels, take mean
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row["label"]


def mixup_data(x, y):
    # blending percentage
    lam = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # data mixing in percentages
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # error as if pred was 100% a and multiply by blending pct
    # then same for b
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train():
    # Setup directories
    esc50_dir = download_esc50_data()

    # Create models directory for saving checkpoints
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    # Setup tensorboard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = models_dir / "tensorboard_logs" / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        ),
        T.AmplitudeToDB(),
        # similar to Dropout for Audio
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80),
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        ),
        T.AmplitudeToDB(),
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="train",
        transform=train_transform,
    )

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="test",
        transform=val_transform,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Use MPS (Metal Performance Shaders) for Apple Silicon, fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,  # 10% of training is spent increasing lr, and rest decr.
    )

    best_accuracy = 0.0

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # Synthetic samples to lower confidence, increase reliability
            # Mix two sounds - data mixing - acts as background noise
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:  # ~70% of time no data mixing
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Validate after each epoch
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                # Pick highest scored class as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)  # Add batch size to running total
                correct += (predicted == target).sum().item()

        accuracy = correct / total * 100
        avg_val_loss = val_loss / len(val_dataloader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        print(
            f"Epoch: {epoch+1} | Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "classes": train_dataset.classes,
                },
                models_dir / "best_model.pth",
            )
            print(f"New best model saved: {accuracy:.2f}%")

    writer.close()
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    train()
