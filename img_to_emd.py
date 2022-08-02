import numpy as np
import h5py
from txt2image_dataset import Text2ImageDataset
import yaml 
import random

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataset=Text2ImageDataset(config['birds_dataset_path'])
print(random.choice(dataset)['right_embed'])