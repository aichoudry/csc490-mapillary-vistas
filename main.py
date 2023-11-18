import torch
import torch.nn.functional as F
import torch.optim as optim
import configparser
import numpy as np
from torch.utils.data import DataLoader
from mapillary_vistas_dataset import MapillaryVistasDataset
from transformations import config_gettransformation
from unet import UNet
from train import train_model
from evaluation import evaluate_model
from torch.utils.data import DataLoader
from plot import plot_graph, plot_results
from os import path

print("Opening config.ini...")

config = configparser.ConfigParser()
config.read("./train.ini")

MAX_IMAGES = config.getint("Hyperparameters", "MaxTrainingImages")
BATCH_SIZE = config.getint("Hyperparameters", "BatchSize")

LEARNING_RATE = config.getfloat("Hyperparameters", "LearningRate")
EPOCHS = config.getint("Hyperparameters", "Epochs")

IMAGE_DIMEN = config.getint("Image Preprocessing", "N")
TRANSFORM_NAME = config.get("Image Preprocessing", "Transformation")
TRANSFORM = config_gettransformation(TRANSFORM_NAME, IMAGE_DIMEN)

EVALUATION_IMAGES = config.get("Validation Plot Images", "Images").split(",")

CHECKPOINT_SAVE_INTERVAL = config.getint("Misc", "SaveCheckpointsInterval")

CRITERION = F.cross_entropy

MODEL_NAME = f"model-{MAX_IMAGES}-{TRANSFORM_NAME}-{IMAGE_DIMEN}-{BATCH_SIZE}-{str(LEARNING_RATE).replace('.', '')}-{EPOCHS}-wCE"

LOAD_CHECKPOINT_PATH = None
SAVE_LOCATION = config.get("Misc", "SaveLocation")
MODEL_LOCATION = path.join(SAVE_LOCATION, MODEL_NAME)

print(F"MAX_IMAGES: {MAX_IMAGES}")
print(F"BATCH_SIZE: {BATCH_SIZE}")
print(F"LEARNING_RATE: {LEARNING_RATE}")
print(F"EPOCHS: {EPOCHS}")
print(F"TRANSFORM: {TRANSFORM_NAME}")
print(F"IMAGE_DIMEN: {IMAGE_DIMEN}")
print(F"EVALUATION_IMAGES: {EVALUATION_IMAGES}")
print(F"CHECKPOINT_SAVE_INTERVAL: {CHECKPOINT_SAVE_INTERVAL}")
print(F"MODEL_LOCATION: {MODEL_LOCATION}")
print(F"MODEL_NAME: {MODEL_NAME}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_dataset = MapillaryVistasDataset(MapillaryVistasDataset.TRAINING,
                                          max_images=MAX_IMAGES,
                                          transform=TRANSFORM)

training_loader = DataLoader(training_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=True, 
                             num_workers=8, 
                             pin_memory=True)


model = UNet(in_channels=3, out_channels=len(MapillaryVistasDataset.color_to_i)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

weights = 1 / np.sqrt(MapillaryVistasDataset.CLASS_DISTRIBUTION) # pc 47
# weights = 1 / np.log(MapillaryVistasDataset.CLASS_DISTRIBUTION + 1) # pc48

if DEVICE == torch.device("cuda"):
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor.cuda()
    model.cuda()

TRAINED_MODEL, TRAINING_LOSSES, TRAINING_ACCURACIES = train_model(model=model, 
                                                                    epochs=EPOCHS, 
                                                                    learning_rate=LEARNING_RATE, 
                                                                    criterion=CRITERION, 
                                                                    device=DEVICE, 
                                                                    name=MODEL_NAME, 
                                                                    checkpoint_path=LOAD_CHECKPOINT_PATH, 
                                                                    epoch_save_interval=CHECKPOINT_SAVE_INTERVAL,
                                                                    save_path=MODEL_LOCATION,
                                                                    weights=weights_tensor,
                                                                    optimizer=optimizer, 
                                                                    training_loader=training_loader)

# Plot the training losses
plot_graph(data=TRAINING_LOSSES, 
           title=f"Training Loss Over {EPOCHS} Epochs", 
           xlabel="Epoch", 
           ylabel="Loss", 
           name="training_loss_graph",
           save_location=MODEL_LOCATION)

plot_graph(data=TRAINING_ACCURACIES, 
           title=f"Training Accuracy Over {EPOCHS} Epochs", 
           xlabel="Epoch", 
           ylabel="Accuracy", 
           name="training_accuracy_graph",
           save_location=MODEL_LOCATION)

# Evaluate the model
TRAINED_MODEL.eval()
validation_dataset = MapillaryVistasDataset(MapillaryVistasDataset.VALIDATION, 
                                            transform=TRANSFORM)
validation_loader = DataLoader(validation_dataset, 
                               batch_size=BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=8, 
                               pin_memory=True)
mIoU_value = evaluate_model(TRAINED_MODEL, validation_loader, DEVICE, MapillaryVistasDataset.NUM_CLASSES)
print(f"Validation mIoU: {mIoU_value:.4f}")

# Image results
print("Testing individual images...")
for image in EVALUATION_IMAGES:
    img_dir = "/virtual/csc490_mapillary/data_v12"
    plot_results(image, TRAINED_MODEL, DEVICE, MODEL_LOCATION)

print("Done")

