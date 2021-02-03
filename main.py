import os
from dataloader import dataloader
from train import train, LogWriter


if __name__ == "__main__":
    # DataLoader
    train_loader = dataloader('./drive/MyDrive/VGG_model/dogs-vs-cats/')

    # Log save
    os.makedirs('./drive/MyDrive/VGG_model/logs/', exist_ok=True)
    writer = LogWriter('./drive/MyDrive/VGG_model/logs/')

    # Training
    train(train_loader, [], writer)
