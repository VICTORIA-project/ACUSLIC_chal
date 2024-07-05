#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
)
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
from PIL import Image

def main():
    # Expansion HP
    x_expansion = 744
    y_expansion = 744
    x_resizing = 256
    y_resizing = 256
    file_format = 'mha'
    new_folder_name = f'full-slice_{x_resizing}x{y_resizing}'
    original_im_dir = repo_path / 'data/original/images/stacked_fetal_ultrasound'
    original_mask_dir = repo_path / 'data/original/masks/stacked_fetal_abdomen'

    # transforms
    preprocess_im = Compose(
            [Resize((x_resizing, y_resizing), interpolation= InterpolationMode.BILINEAR),]
    )
    preprocess_label = Compose(
            [Resize((x_resizing, y_resizing), interpolation= InterpolationMode.NEAREST),]
    )
    low_res_trans = Compose(
            [Resize((x_resizing//4, y_resizing//4), interpolation= InterpolationMode.NEAREST),]
    )

    # new images and labels
    save_dir = repo_path / 'data/preprocessed' / new_folder_name
    im_dir = save_dir / f'images_{file_format}'
    label_dir = save_dir / f'masks_{file_format}'
    save_dir.mkdir(exist_ok=True)
    im_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    metadata = pd.read_csv(repo_path / 'data/original/metadata.csv')

    counter = tqdm(total=len(metadata))
    for _, ex_row in metadata.iterrows():
        # read images and labels
        ex_name = ex_row['uuid'] + '.' + 'mha'
        image_path = original_im_dir / ex_name
        label_path = original_mask_dir / ex_name
        assert image_path.exists() and label_path.exists()
        # get image and label
        im = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # now, we complete the images and labels to the expansion variables
        if im.shape[2]<x_expansion:
            im = np.concatenate((im, np.zeros((im.shape[0], im.shape[1], x_expansion-im.shape[2]), dtype=np.int8)), axis=2)
            label = np.concatenate((label, np.zeros((label.shape[0], label.shape[1], x_expansion-label.shape[2]), dtype=np.int8)), axis=2)

        if im.shape[1]<y_expansion:
            # print('Expanding y dimension')
            im = np.concatenate((im, np.zeros((im.shape[0], y_expansion-im.shape[1], im.shape[2]), dtype=np.int8)), axis=1)
            label = np.concatenate((label, np.zeros((label.shape[0], y_expansion-label.shape[1], label.shape[2]), dtype=np.int8)), axis=1)

        # go through all z values (framkes)
        for z in range(im.shape[0]):
            # preprocess image
            im_slice = Image.fromarray(im[z])
            im_slice = preprocess_im(im_slice)
            im_slice = np.asarray(im_slice)
            # put channel first and repeat in RGB
            im_slice = np.repeat(np.expand_dims(im_slice, axis=0), 3, axis=0)
            im_slice = im_slice.astype(np.int32)

            # preprocess label
            label_slice = Image.fromarray(label[z])
            label_slice = preprocess_label(label_slice)
            label_slice = np.asarray(label_slice).astype(np.int32)
            # save plane type
            plane_type = label_slice.max()
            # send label value to 1
            label_slice[label_slice>0] = 1

            # saving path
            save_name = ex_name.replace('.mha', f'_z{z}_plane_{plane_type}.{file_format}')
            # save image
            sitk.WriteImage(sitk.GetImageFromArray(im_slice), str(im_dir / save_name))
            # save label
            sitk.WriteImage(sitk.GetImageFromArray(label_slice), str(label_dir / save_name))
        counter.update(1)

if __name__ == '__main__':
    main()