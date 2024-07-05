from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    Resized,
)
   
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
