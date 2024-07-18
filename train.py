import torch
import torch.nn as nn
from pathlib import Path
import os
from monai.transforms import (
    Compose,
    ScaleIntensity,
    RandGaussianNoise,
    RandGaussianSmooth,
    EnsureType
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from timeit import default_timer as timer
from src.models.frame_detector import build_models
from src.datasets.acouslic_dataset import AcouslicDatasetFull

this_path = Path().resolve()
repo_path = Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path):  # while not in the root of the repo
    repo_path = repo_path.parent  #go up one level

data_path = repo_path / 'data' / 'acouslic-ai-train-set'
assert data_path.exists()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_weights(metadata_path: Path):
    df = pd.read_csv(metadata_path)
    labels = df['plane_type']
    weights = compute_class_weight('balanced', classes=[0, 1, 2], y=labels)
    return torch.Tensor(weights)


def main():
    batch_size = 2
    hidden_dim = 768
    lr = 1e-4
    weights = torch.Tensor([0.3441, 32.8396, 15.6976])

    # create dataset
    metadata_path = data_path / 'circumferences/fetal_abdominal_circumferences_per_sweep.csv'
    df = pd.read_csv(metadata_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold_n, (train_ids, val_ids) in enumerate(kf.split(df.subject_id.unique())):
        # print(train_ids, val_ids)
        print(len(train_ids), len(val_ids))
        break

    preproc_transforms = Compose([
                        ScaleIntensity(),
                        EnsureType()
                    ])
    # frame_transforms = Compose([
    #             RandGaussianSmooth(prob=0.1),
    #             RandGaussianNoise(prob=0.5),
    #             ])
    train_dataset = AcouslicDatasetFull(metadata_path=metadata_path,
                                        preprocess_transforms=preproc_transforms,
                                        subject_ids=train_ids)
    val_dataset = AcouslicDatasetFull(metadata_path=metadata_path,
                                      preprocess_transforms=preproc_transforms,
                                      subject_ids=val_ids)

    def my_collate_fn(batch):
        return batch
    
    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=my_collate_fn)
    val_dl = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=my_collate_fn)

    # create models and move to device
    encoder, projector, transformer, classifier = build_models(hidden_dim)
    encoder.to(DEVICE)
    projector.to(DEVICE)
    transformer.to(DEVICE)
    classifier.to(DEVICE)

    # optimizer
    params = list(projector.parameters()) + list(transformer.parameters()) \
        + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    NUM_EPOCHS = 20

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(encoder,
                                 projector,
                                 transformer,
                                 classifier,
                                 optimizer,
                                 criterion,
                                 train_dl)
        end_time = timer()
        val_loss = evaluate(encoder,
                            projector,
                            transformer,
                            classifier,
                            criterion,
                            val_dl)

        print((f'Epoch: {epoch}, Train loss: {train_loss:.3f}, '
               f'Val loss: {val_loss:.3f}, '
               f'Epoch time (total) = {(end_time - start_time):.3f}s'))


def train_epoch(encoder,
                projector,
                transformer,
                classifier,
                optimizer,
                criterion,
                dataloader):

    projector.train()
    transformer.train()
    classifier.train()

    losses = 0
    n_frames = 0

    for samples in tqdm(dataloader, total=len(list(dataloader))):

        # model fwd
        encodings = []
        labels = []
        for sample in samples:
            image = sample['image'].unsqueeze(1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = encoder(image)
            output = projector(output)
            encodings.append(output)
            labels.append(sample['labels'])

        encodings, masks = transformer(encodings)

        # classifier
        encodings = encodings[~masks, :]
        labels = torch.cat(labels)
        logits = classifier(encodings)

        optimizer.zero_grad()

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        n_frames += len(labels)

    return losses / n_frames


def evaluate(encoder,
             projector,
             transformer,
             classifier,
             criterion,
             dataloader):

    projector.eval()
    transformer.eval()
    classifier.eval()

    losses = 0
    n_frames = 0

    for samples in tqdm(dataloader, total=len(list(dataloader))):

        # model fwd
        encodings = []
        labels = []
        for sample in samples:
            image = sample['image'].unsqueeze(1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = encoder(image)
            output = projector(output)
            encodings.append(output)
            labels.append(sample['labels'])

        encodings, masks = transformer(encodings)

        # classifier
        encodings = encodings[~masks, :]
        labels = torch.cat(labels)
        logits = classifier(encodings)

        loss = criterion(logits, labels)

        losses += loss.item()
        n_frames += len(labels)

    return losses / n_frames


if __name__ == '__main__':
    main()