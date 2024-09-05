import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from custom_dc import CocoDataset

def get_model(num_classes):
    print("Loading Faster R-CNN ResNet50 FPN model with COCO weights...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"Model loaded with {num_classes} classes.")
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    print(f"Starting epoch {epoch + 1}...")
    for i, (images, targets) in enumerate(data_loader):
        print(f"Processing batch {i + 1}/{len(data_loader)}...")
        
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Batch {i + 1} processed with loss: {losses.item()}")
    print(f"Epoch {epoch + 1} completed.\n")

def main():
    base_dir = 'data'
    annotation_file = 'annotations/coco_annotations.json'
    num_classes = len(os.listdir(os.path.join(base_dir, 'Train'))) + 1
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.01

    print("Loading dataset...")
    dataset = CocoDataset(root=base_dir, annotation_file=annotation_file, transforms=ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    print(f"Dataset loaded with {len(dataset)} samples. Batch size: {batch_size}")

    print("Setting up model and optimizer...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model = get_model(num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    print("Model and optimizer setup completed.\n")

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

    print("Saving trained model...")
    torch.save(model.state_dict(), 'fasterrcnn_model.pth')
    print("Model saved as 'fasterrcnn_model.pth'.")

if __name__ == "__main__":
    main()