from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    Resize,
    SpatialPad,
)
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, List

   
class Acouslic_dataset(Dataset):
    
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in list_dir[0]]
        labels = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in list_dir[1]]
        # all samples
        self.sample_list = list(zip(images,labels))
        # low_resolution transform
        self.resize=Compose([Resized(keys=["label"], spatial_size=(64, 64),mode=['nearest'])])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        if self.transform:
            sample=self.transform({"image": self.sample_list[idx][0], "label": self.sample_list[idx][1]})
        
        sample['low_res_label']=self.resize({"label":sample['label']})['label'][0]
        sample['label']=sample['label'][0]
        return sample


class AcouslicDatasetFull(Dataset):
    
    def __init__(self, metadata_path: Path,
                 transforms=None,
                 subject_ids: Optional[List[int]] = None):
        
        self.transform = transforms  # using transform in torch!
        self.images_path = metadata.parent.parent / 'images/stacked_fetal_ultrasound'
        self.masks_path = metadata.parent.parent / 'masks/stacked_fetal_abdomen'

        self.preprocess_image = Compose([
            SpatialPad(spatial_size=(-1, 744, 744),
                       method='symmetric',
                       mode='constant', constant_values=0),
            Resized(spatial_size=(-1, 256, 256), mode='bilinear')
        ])
        self.preprocess_mask = Compose([
            SpatialPad(spatial_size=(-1, 744, 744),
                       method='symmetric',
                       mode='constant', constant_values=0),
            Resized(spatial_size=(-1, 256, 256), mode='nearest')
        ])
        metadata = pd.read_csv(metadata_path)
        if subject_ids is not None:
            metadata = metadata[metadata['subject_id'].isin(subject_ids)]
        else:
            subject_ids = metadata['subject_id'].unique()
            train_ids, val_ids = train_test_split(subject_ids,
                                                  test_size=0.2,
                                                  random_state=0,
                                                  shuffle=True)
            metadata = metadata[metadata['subject_id'].isin(train_ids)]
        
        self.subject_ids = subject_ids
        self.train_ids = subject_ids if subject_ids is not None else train_ids
        self.val_ids = None if subject_ids is not None else val_ids
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        uuid = self.metadata['uuid'][idx]
        img_name = f'{uuid}.mha'
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.images_path / img_name))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(self.masks_path / img_name))

        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        if self.transform:
            sample=self.transform({"image": self.sample_list[idx][0], "label": self.sample_list[idx][1]})
        
        sample['low_res_label']=self.resize({"label":sample['label']})['label'][0]
        sample['label']=sample['label'][0]
        return sample