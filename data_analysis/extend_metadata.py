# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def main():
    # using the circumference mesuarement csv as baseline
    meta_path = repo_path / 'data/original/circumferences/fetal_abdominal_circumferences_per_sweep.csv'
    metadata = pd.read_csv(meta_path)
    # directories
    mask_dir = 'data/original/masks/stacked_fetal_abdomen'

    # new columns 'z_values_opt' and 'z_values_subopt'
    metadata['z_values_opt'] = None
    metadata['z_values_subopt'] = None

    counter = tqdm(total=len(metadata))  # initiate the progress bar
    # read the first row
    for i, ex_row in metadata.iterrows():
        mask_path = repo_path / mask_dir / f'{ex_row["uuid"]}.mha'
        assert mask_path.exists(), 'The example does not exist'
        # read mask
        mask = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask)
        # get frames where mask is not empty
        z_values_opt = np.unique(np.where(mask_array==1)[0])
        z_values_subopt = np.unique(np.where(mask_array==2)[0])
        # add columns to the series
        ex_row['z_values_opt'] = len(z_values_opt)
        ex_row['z_values_subopt'] = len(z_values_subopt)
        # update the row
        metadata.iloc[i] = ex_row
        counter.update(1)  # update the progress bar

    # save the updated metadata
    metadata.to_csv(repo_path / 'data/original' / 'metadata.csv', index=False)



if __name__ == '__main__':
    main()