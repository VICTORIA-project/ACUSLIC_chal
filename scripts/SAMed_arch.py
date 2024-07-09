#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

# Libraries
from sklearn.model_selection import KFold
import numpy as np
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
import wandb
import logging
# accelerate
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
# torch
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
# others
from importlib import import_module
from sklearn.metrics import jaccard_score
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAffined,
    OneOf,
)

# extra imports
from scripts.datasets import Acouslic_dataset
sys.path.append(str(repo_path / 'SAMed'))
from SAMed.segment_anything import sam_model_registry
from SAMed.utils import DiceLoss

class SAMed_segmentor():

    def __init__(self, config_path:Path):

        # define args from the config file
        config_path = repo_path / config_path
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.args = argparse.Namespace(**config)
        # read metadata
        self.metadata = pd.read_csv(repo_path / 'data/original/metadata.csv')

        # important paths
        self.checkpoint_dir = repo_path / 'checkpoints' # SAMed checkpoints
        self.data_path = repo_path / self.args.data_path # path to data


    def init_sam(self):
        """defines the sam model and loads the weights (and pretrained if specified)
        attributes:
        model: the model object
        """
        sam, _ = sam_model_registry['vit_b'](image_size=self.args.image_size,
                                            num_classes=self.args.num_classes,
                                            checkpoint=str(self.checkpoint_dir / 'sam_vit_b_01ec64.pth'),
                                            pixel_mean=[0, 0, 0],
                                            pixel_std=[1, 1, 1])
        # load lora model
        pkg = import_module('sam_lora_image_encoder')
        model = pkg.LoRA_Sam(sam, 4) # lora rank is 4
        if self.args.pretrained_path: # load pretrained weights
            model.load_lora_parameters(str(repo_path / self.args.pretrained_path))
        
        self.model = model

    def define_transformations(self):
        """When called, defines train and validation transformations.
        attributes:
        train_transform: Compose object for train transformations
        val_transform: Compose object for validation transformations
        """
        deform = Rand2DElasticd(
            keys=["image", "label"],
            prob=0.5,
            spacing=(7, 7),
            magnitude_range=(1, 2),
            rotate_range=(np.pi / 6,),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            mode=['bilinear','nearest']
        )

        affine = RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi / 6),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            mode=['bilinear','nearest']

        )

        self.train_transform = Compose(
            [
                EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'), # to add one channel dim to the label (1,256,256)

                ScaleIntensityd(keys=["image"]), # to scale the image intensity to [0,1]

                ## train-specific transforms
                RandRotated(keys=["image", "label"], range_x=(-np.pi / 12, np.pi / 12), prob=0.5, keep_size=True,mode=['bilinear','nearest']),

                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),

                RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5,mode=['area','nearest']),

                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                RandGaussianNoised(keys=["image"], mean=0, std=0.1, prob=0.5),

                # TODO check this
                OneOf(transforms=[affine, deform], weights=[0.8, 0.2]), # apply one of the two transforms with the given weights
                ##

                Resized(keys=["image", "label"], spatial_size=(self.args.image_size, self.args.image_size),mode=['area','nearest']),

                EnsureTyped(keys=["image"] ), # ensure it is a torch tensor or np array
            ]
        )
        self.train_transform.set_random_state(seed=self.args.training_seed) # set the seed for the transforms

        self.val_transform = Compose(
            [
                EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),

                ScaleIntensityd(keys=["image"]),

                Resized(keys=["image", "label"], spatial_size=(self.args.image_size, self.args.image_size),mode=['area','nearest']),
                EnsureTyped(keys=["image"])
            ])
    
    def define_fold(self):
        # define fold type
        kf = KFold(n_splits=self.args.num_folds,shuffle=self.args.split_shuffle,random_state=self.args.split_seed)

        return kf
        
    def load_data(self):
        # get all files in the images and labels folders
        image_files = np.array([str(i) for i in (self.data_path / 'images_mha').rglob("*.mha")]) 
        label_files = np.array([str(i) for i in (self.data_path / 'masks_mha').rglob("*.mha")])

        # get the file name (uuid) of all the train subjects
        train_subjects = self.metadata[self.metadata['subject_id'].isin(self.metadata['subject_id'].unique()[train_ids])] # extract training subjects based on id
        train_file_name = train_subjects['uuid'].unique() # get uuids of train subjects
        train_images = [file for file in image_files if any(f'{name_file}' in file for name_file in train_file_name)]
        train_labels = [file for file in label_files if any(f'{name_file}' in file for name_file in train_file_name)]
        list_train = [train_images, train_labels]

        # validation
        val_subjects = self.metadata[self.metadata['subject_id'].isin(self.metadata['subject_id'].unique()[val_ids])]
        val_file_name = val_subjects['uuid'].unique()
        val_images = [file for file in image_files if any(f'{name_file}' in file for name_file in val_file_name)]
        val_labels = [file for file in label_files if any(f'{name_file}' in file for name_file in val_file_name)]
        list_val = [val_images, val_labels]

        # define datasets and dataloaders
        db_train = Acouslic_dataset(transform=self.train_transform,list_dir=list_train)
        db_val = Acouslic_dataset(transform=self.val_transform,list_dir=list_val)

        trainloader = DataLoader(db_train, batch_size=self.args.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valloader = DataLoader(db_val, batch_size=self.args.val_batch_size, shuffle=True, num_workers=8, pin_memory=True)