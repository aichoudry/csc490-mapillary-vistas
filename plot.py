import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from os import makedirs, path
from torchvision import transforms
from PIL import Image
from mapillary_vistas_dataset import MapillaryVistasDataset


def plot_graph(data, title, xlabel, ylabel, name, save_location):
    # Plot the training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig(path.join(save_location, f"{name}.png"))


def label_to_rgb(mask_tensor, color_map):
    # Create an empty RGB image
    rgb_image = torch.zeros(3, mask_tensor.shape[-2], mask_tensor.shape[-1], dtype=torch.uint8)

    # Map each label to its RGB color
    for label, color in color_map.items():
        for channel, intensity in enumerate(color):
            rgb_image[channel][mask_tensor == label] = intensity

    return rgb_image

def plot_results(image, model, device, save_location):
    input_image_path, ground_truth_image_path = MapillaryVistasDataset.get_validation_image_paths(image)

    color_map = MapillaryVistasDataset.i_to_color
    crop_size = 1024
    input_image = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    _, predicted_mask = torch.max(prediction, 1)

    # Convert tensor to color-coded image
    predicted_rgb = label_to_rgb(predicted_mask.squeeze(0).cpu(), color_map)
    predicted_image = transforms.ToPILImage()(predicted_rgb)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display input image

    transform = transforms.Compose([transforms.CenterCrop(crop_size)])

    axes[0].imshow(transform(input_image))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Display predicted mask
    axes[1].imshow((predicted_image))
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    # Display ground truth, if provided
    if ground_truth_image_path:
        ground_truth = Image.open(ground_truth_image_path)
        axes[2].imshow(transform(ground_truth), cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    else:
        axes[2].remove()

    plt.tight_layout()
    results_directory = path.join(save_location, "image_results")
    makedirs(results_directory, exist_ok=True)

    plt.savefig(path.join(results_directory, f"{image}.png"))