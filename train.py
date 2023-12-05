import torch
import os
from datetime import datetime


def train_model(model, epochs, learning_rate, criterion, training_loader, optimizer, device, name, save_path,
                weights=None, 
                epoch_save_interval=10, 
                checkpoint_path=None):
    accuracies = []
    losses = []

    os.makedirs(save_path, exist_ok=True)

    checkpoints_directory = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_directory, exist_ok=True)

    model.train()
    
    print(f'Training {name}...')
    print(f'epochs={epochs} lr={learning_rate}, error={criterion}')

    start_epoch = 0
    if checkpoint_path is not None:
        print(f'Attempting to load checkpoint={checkpoint_path}...')
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(training_loader)
        interval = int(total_batches * 0.30)
        if interval == 0:
            interval = 1

        for batch_index, (images, masks) in enumerate(training_loader):
            if batch_index % interval == 0:
                print(f"    [{epoch}] Processed {batch_index}/{total_batches} batches ({(batch_index/total_batches)*100:.2f}%)")

            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks, weight=weights)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += masks.numel()  # assuming masks are of shape (batch_size, height, width)
            correct += (predicted == masks).sum().item()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(training_loader.dataset)
        losses.append(epoch_loss)

        epoch_accuracy = 100 * correct / total
        accuracies.append(epoch_accuracy)

        # Save checkpoint at the end of each epoch
        if epoch % epoch_save_interval == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_save = os.path.join(checkpoints_directory, checkpoint_filename)
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': accuracies,
                    'accuracy': accuracies
                }, checkpoint_save)
                print(f"Checkpoint {epoch+1} saved to {checkpoint_save}")
            except:
                print(f"Error: Couldn't save {checkpoint_filename} (likely no space)...")
        
        formatted_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{formatted_time}] Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")

    result = {
        'state_dict': model.state_dict(),
        'accuracies': accuracies,
        'losses': losses
    }
    torch.save(result, os.path.join(save_path, f'{name}.pth'))
    
    return model, losses, accuracies

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        epoch_number = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return epoch_number
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch")
        return 0
    