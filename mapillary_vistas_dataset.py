import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import json
import torch

ROOT_DIR = "/virtual/csc490_mapillary/data_v12"

def load_config(root_dir: str):
  f = open(f'{root_dir}/config.json')
  data = json.load(f)
  data = data['labels']
  d1 = {}
  d2 = {}
  d3 = {}
  for i, label in enumerate(data):
    color = tuple(label['color'])
    d1[color] = label['readable']
    d2[color] = i
    d3[i] = color
  return d1, d2, d3

class MapillaryVistasDataset(Dataset):
  color_name, color_to_i, i_to_color = load_config(ROOT_DIR)
  NUM_CLASSES = len(color_to_i)
  
  TRAINING = "TRAINING"
  TESTING = "TESTING"
  VALIDATION = "VALIDATION"

  config = {
    TRAINING: [
        ROOT_DIR + '/training/images',
        ROOT_DIR + '/training/labels'
    ],
    VALIDATION: [
        ROOT_DIR + '/validation/images',
        ROOT_DIR + '/validation/labels'
    ],
    TESTING: [
        ROOT_DIR + '/testing/images',
        ROOT_DIR + '/testing/labels'
    ],
  }

  def __init__(self, dataset=TRAINING, max_images=None, transform=None, mask_transform=None):
    if dataset not in self.config:
        raise(f"Invalid Dataset. Please enter one of {list(self.config.keys())}")

    self.directory = ROOT_DIR
    self.image_dir = self.config[dataset][0]
    self.mask_dir = self.config[dataset][1]

    self.image_files = sorted(os.listdir(self.image_dir))[:max_images]
    self.mask_files = sorted(os.listdir(self.mask_dir))[:max_images]
    if dataset == self.TESTING:
        print(self.mask_dir, os.listdir(self.mask_dir))

    self.transform = transform
    self.mask_transform = mask_transform

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = os.path.join(self.image_dir, self.image_files[idx])
    mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

    image = Image.open(img_name).convert('RGB')
    mask = Image.open(mask_name).convert('RGB')

    if self.transform:
        image = self.transform(image)
    if self.mask_transform:
        mask = self.mask_transform(mask)

    mask_array = np.array(mask)
    mask_label = np.zeros(mask_array.shape[:2], dtype=np.compat.long)

    for rgb, label in self.color_to_i.items():
        mask_label[(mask_array == np.array(rgb).reshape(1, 1, 3)).all(2)] = label

    mask = torch.from_numpy(mask_label)

    return image, mask