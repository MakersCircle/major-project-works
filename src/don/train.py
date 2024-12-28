import re
import cv2
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_xla
import torch_xla.core.xla_model as xm
from torchvision.models.video import swin3d_t
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, ToPILImage
class AccidentDataset(Dataset):
    def __init__(self, annotation_file, video_dir, sequence_length=32, transform=None):
        """
        Args:
            annotation_file (str): Path to the annotations file (e.g., Crash-1500.txt).
            video_dir (str): Directory containing the video files.
            sequence_length (int): Number of frames in each input sequence.
            transform (callable, optional): A function/transform to apply to video frames.
        """
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Load and parse annotations
        self.data = []

        self.to_pil = ToPILImage()


        with open(annotation_file) as f:
            for line in f.readlines():
                # Use regex to extract fields properly
                match = re.match(r"^(\d+),(\[.*?\]),(\d+),(\d+),(Day|Night),(Normal|Snowy|Rainy),(Yes|No)$", line.strip())
                if match:
                    vidname = match.group(1)
                    binlabels = eval(match.group(2))  # Safely evaluate the binary labels
                    startframe = int(match.group(3))
                    youtubeID = match.group(4)
                    timing = match.group(5)
                    weather = match.group(6)
                    egoinvolve = match.group(7)

                    self.data.append({
                        'vidname': vidname,
                        'binlabels': binlabels,
                        'startframe': startframe,
                        'youtubeID': youtubeID,
                        'timing': timing,
                        'weather': weather,
                        'egoinvolve': egoinvolve
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        vidname = sample['vidname']
        binlabels = sample['binlabels']

        # Load video frames
        video_path = video_dir / f'{vidname}.mp4'
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            success, frame = cap.read()
        cap.release()

        # Ensure we have enough frames
        if len(frames) < self.sequence_length:
            raise ValueError(f"Video {vidname} has fewer than {self.sequence_length} frames.")

        # Select the last `sequence_length` frames and labels
        frames = frames[-self.sequence_length:]
        labels = binlabels[-self.sequence_length:]

        frames = [self.to_pil(frame) for frame in frames]

        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # Convert list of tensors to a 4D tensor
        frames = frames.permute(1, 0, 2, 3)
        labels = torch.tensor(labels, dtype=torch.float32)

        return frames, labels

    # Define transforms
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_path = Path().resolve().parent / 'datasets' / 'CCD'
    annotation_file = dataset_path / 'Crash-1500.txt'
    video_dir = dataset_path / 'videos'
    sequence_length = 50
    # Create the dataset
    accident_dataset = AccidentDataset(annotation_file=annotation_file,
                                       video_dir=video_dir,
                                       sequence_length=sequence_length,
                                       transform=transform)
    # Create the dataloader
    dataloader = DataLoader(accident_dataset, batch_size=8, shuffle=True, num_workers=4)
    # Example usage
    for batch_frames, batch_labels in dataloader:
        print("Batch frames shape:", batch_frames.shape)  # Expected: [batch_size, 32, 3, 224, 224]
        print("Batch labels shape:", batch_labels.shape)  # Expected: [batch_size, 32]
        break
    model = swin3d_t(weights="KINETICS400_V1")
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 1),  # Predict probability of accident per frame
        nn.Sigmoid()  # Output probabilities in [0, 1]
    )
    device = xm.xla_device()
    model = model.to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for each frame
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Iterate through batches
        for frames, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            frames, labels = frames.to(device), labels.to(device)

            # Expand labels to match the model output shape
            labels = labels.unsqueeze(2)  # Shape: [batch_size, sequence_length, 1]

            # Forward pass
            outputs = model(frames)  # Shape: [batch_size, sequence_length, 1]
            outputs = outputs.squeeze(2)  # Shape: [batch_size, sequence_length]

            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        # Print average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), "swin3d_finetuned.pth")