import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mapillary_vistas_dataset import MapillaryVistasDataset

MAX_IMAGES = 2000
IMAGE_DIMEN = 256
BATCH_SIZE = 16

IMAGE_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(IMAGE_DIMEN),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Keep this always
])
MAKS_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(IMAGE_DIMEN)
])

TRAINING_DATASET_NAME = f"{MAX_IMAGES}-{IMAGE_DIMEN}-{BATCH_SIZE}-crop"
training_dataset = MapillaryVistasDataset(MapillaryVistasDataset.TRAINING,
                                          max_images=MAX_IMAGES,
                                          transform=IMAGE_TRANSFORM,
                                          mask_transform=MAKS_TRANSFORM)
training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

from unet import UNet
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from train import train_model

LEARNING_RATE = 0.001
EPOCHS = 50
CRITERION = F.cross_entropy

LOAD_CHECKPOINT_PATH = None

MODEL_NAME = f"unet-{TRAINING_DATASET_NAME}-{EPOCHS}-{str(LEARNING_RATE).replace('.', '')}"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=3, out_channels=len(MapillaryVistasDataset.color_to_i)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


LOAD_CHECKPOINT_PATH = None

model_info = train_model(model=model, 
                                 epochs=EPOCHS, 
                                 learning_rate=LEARNING_RATE, 
                                 criterion=CRITERION, 
                                 training_loader=training_loader, 
                                 optimizer=optimizer, 
                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                 name=MODEL_NAME, 
                                 checkpoint_path=LOAD_CHECKPOINT_PATH, 
                                 save=True)



from evaluation import evaluate_model
from mapillary_vistas_dataset import MapillaryVistasDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

SAVED_MODEL_NAME = MODEL_NAME #"unet-256-crop-50-2000-001"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if SAVED_MODEL_NAME:
    saved_model_path = f"/virtual/csc490_mapillary/models/{SAVED_MODEL_NAME}/{SAVED_MODEL_NAME}.pth"
    print(f"Evaluating {saved_model_path}...")
    model = UNet(in_channels=3, out_channels=MapillaryVistasDataset.NUM_CLASSES).to(device)
    model_info = torch.load(saved_model_path)
    model.load_state_dict(model_info['state_dict'])
    print(f"Accuracies: {model_info['accuracies']}")
    print(f"Losses: {model_info['losses']}")
else:
    print(f"Evaluating {SAVED_MODEL_NAME}..")

model.eval()

validation_dataset = MapillaryVistasDataset(MapillaryVistasDataset.VALIDATION,
                                            transform=IMAGE_TRANSFORM,
                                            mask_transform=MAKS_TRANSFORM)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
mIoU_value = evaluate_model(model, 
                            validation_loader, 
                            device, 
                            MapillaryVistasDataset.NUM_CLASSES)
print(f"Validation mIoU: {mIoU_value:.4f}")
