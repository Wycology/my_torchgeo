import torch
import torchseg
import pandas as pd
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader
from torchgeo.transforms import AppendNDVI
from albumentations.pytorch import ToTensorV2
from torchgeo.datasets import RasterDataset, VectorDataset
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define augmentation pipeline
augmentation = A.Compose([
    A.RandomRotate90(p = 0.5),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.5),
])

append_ndvi = AppendNDVI(index_nir=6, index_red=2)

def stack_samples(batch):
    # Extract images and labels from the batch
    images = [torch.nan_to_num(sample['image'], nan=0.0) for sample in batch]  # Replace NaN with 0 in images
    labels = [sample['mask'] for sample in batch]

    augmented_images = []
    augmented_labels = []

    for img, lbl in zip(images, labels):

      img = append_ndvi(img).squeeze(0)

      img_np = img.permute(1, 2, 0).numpy()
      lbl_np = lbl.squeeze(0).numpy()

      augmented = augmentation(image = img_np, mask = lbl_np)

      augmented_images.append(torch.tensor(augmented["image"]).permute(2, 0, 1))
      augmented_labels.append(torch.tensor(augmented["mask"]).unsqueeze(0))

    # Stack the images and labels into batch tensors
    images = torch.stack(augmented_images)
    labels = torch.stack(augmented_labels).squeeze(1).long()

    return images, labels

# Datasets
raster_dataset = RasterDataset(paths="kenya.tif")
vector_dataset = VectorDataset(paths='kenya_label.gpkg', label_name='tea_no_tea')
vector_dataset.is_image = False
dataset = raster_dataset & vector_dataset

# Samplers
train_sampler = RandomGeoSampler(dataset=dataset, size=32, length=1000)
val_sampler = RandomGeoSampler(dataset=dataset, size=32, length=300)
test_sampler = RandomGeoSampler(dataset=dataset, size=32, length=200)
pred_sampler = GridGeoSampler(dataset=dataset, size=32, stride = 16)

# DataLoader
train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size = 16, collate_fn=stack_samples)
val_loader = DataLoader(dataset=dataset, sampler=val_sampler, batch_size = 16, collate_fn=stack_samples)
test_loader = DataLoader(dataset=dataset, sampler=test_sampler, batch_size = 16, collate_fn=stack_samples)
pred_loader = DataLoader(dataset=dataset, sampler=pred_sampler, batch_size = 16, collate_fn=stack_samples)

# Model
model = torchseg.Unet(
    encoder_name="resnet18",
    encoder_weights=False,
    in_channels=11,
    classes=2
).to(device)

# Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Metrics
def compute_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)  # Get the class predictions
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

def compute_iou(output, labels, num_classes=2):
    _, preds = torch.max(output, dim=1)  # Get the class predictions
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union == 0:
            ious.append(float('nan'))  # If union is 0, skip the IoU for this class
        else:
            ious.append(intersection / union)
    mean_iou = torch.nanmean(torch.tensor(ious))  # Mean IoU, ignoring NaNs
    return mean_iou.item()

# Initialize a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy", "train_iou", "val_loss", "val_accuracy", "val_iou"])

# Saving the best model
best_val_iou = 0.0
best_epoch = 0
best_model_path = "best_model.pth"  # Path to save the best model

# Training loop
epochs = 50  # Number of epochs
for epoch in range(epochs):
    # Training phase
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_accuracy = 0.0
    running_iou = 0.0
    batch_count = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} [Train]", dynamic_ncols=True, leave=True, position=0) as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            output = model(images)

            # Calculate loss
            loss = loss_fn(output, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

            # Update metrics
            running_loss += loss.item()
            running_accuracy += compute_accuracy(output, labels)
            running_iou += compute_iou(output, labels)
            batch_count += 1

            # Update progress bar
            avg_loss = running_loss / batch_count
            avg_accuracy = running_accuracy / batch_count
            avg_iou = running_iou / batch_count
            pbar.set_postfix(loss=f"{avg_loss:.4f}", accuracy=f"{avg_accuracy:.2%}", iou=f"{avg_iou:.4f}")
            pbar.update(1)

    # Print training summary
    print(f"Training - Epoch {epoch + 1}/{epochs}: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2%}, Avg IoU: {avg_iou:.4f}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_accuracy = 0.0
    val_iou = 0.0
    val_batch_count = 0

    with torch.no_grad():  # Disable gradient computation for validation
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{epochs} [Validation]", dynamic_ncols=True, leave=True, position=0) as pbar:
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                output = model(images)

                # Calculate loss
                loss = loss_fn(output, labels)

                # Update metrics
                val_loss += loss.item()
                val_accuracy += compute_accuracy(output, labels)
                val_iou += compute_iou(output, labels)
                val_batch_count += 1

                # Update progress bar
                avg_val_loss = val_loss / val_batch_count
                avg_val_accuracy = val_accuracy / val_batch_count
                avg_val_iou = val_iou / val_batch_count
                pbar.set_postfix(loss=f"{avg_val_loss:.4f}", accuracy=f"{avg_val_accuracy:.2%}", iou=f"{avg_val_iou:.4f}")
                pbar.update(1)

    # Check if the current epoch has the best validation iou

    if avg_val_iou > best_val_iou:
      best_val_iou = avg_val_iou
      best_epoch = epoch + 1

      # Save the model state
      torch.save(model.state_dict(), best_model_path)
      print(f"New best model saved at epoch: {best_epoch} with validation IoU of: {best_val_iou:.4f}")

    # Print validation summary
    print(f"Validation - Epoch {epoch + 1}/{epochs}: Avg Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_val_accuracy:.2%}, Avg IoU: {avg_val_iou:.4f}")

    # Append metrics to DataFrame
     # Append metrics to the DataFrame
    new_row = pd.DataFrame([{
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_accuracy": avg_accuracy,
        "train_iou": avg_iou,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "val_iou": avg_val_iou
    }])

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    # Save metrics to a CSV file
    metrics_df.to_csv("training_metrics.csv", index=False)
    print("Metrics saved to 'training_metrics.csv'")