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
import wandb
from torchmetrics.classification import Accuracy
from datetime import datetime
import sys

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
    NUM_EPOCHS = 20
    batch_size = 2
    hidden_dim = 768
    lr = 1e-4
    weights = torch.Tensor([0.3441, 32.8396, 15.6976]).to(DEVICE)
    workers = 2
    exp_name = f'{sys.argv[1]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    ckpt_path = repo_path / f'checkpoints/{exp_name}'
    ckpt_path.mkdir(parents=True, exist_ok=True)
    config = {
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "lr": lr,
            "loss_weights": weights,
            "n_workers": workers,
            "device": DEVICE,
            "ckpt_path": ckpt_path,
            "exp_name": exp_name,
            "num_epochs": NUM_EPOCHS
        }
    run = wandb.init(
            name=exp_name,
            project="frame-detection",
            tags=["baseline"],
            config=config,
            settings=wandb.Settings(code_dir=".")
        )
    print('Experiment name: ', exp_name)
    # create dataset
    metadata_path = data_path / 'circumferences/fetal_abdominal_circumferences_per_sweep.csv'
    df = pd.read_csv(metadata_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold_n, (train_ids, val_ids) in enumerate(kf.split(df.subject_id.unique())):
        # print(train_ids, val_ids)
        break

    preproc_transforms = Compose([
                        ScaleIntensity()
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
                          num_workers=workers,
                          collate_fn=my_collate_fn)
    val_dl = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=workers,
                        collate_fn=my_collate_fn)
    print(f'Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}')
    # create models and move to device
    encoder, projector, transformer, classifier = build_models(hidden_dim)
    encoder.to(DEVICE)
    projector.to(DEVICE)
    transformer.to(DEVICE)
    classifier.to(DEVICE)
    print('Models created and moved to device')

    # optimizer
    params = list(projector.parameters()) + list(transformer.parameters()) \
        + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    NUM_EPOCHS = 20
    print('Starting training')
    min_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss, train_acc_avg = train_epoch(encoder,
                                                projector,
                                                transformer,
                                                classifier,
                                                optimizer,
                                                criterion,
                                                train_dl)
        end_time = timer()
        wandb.log({"epoch": epoch,
                   "train_acc/epoch": train_acc_avg.item(),
                   "train_loss/epoch": train_loss})
        val_loss, val_acc_avg, val_acc = evaluate(encoder,
                                                  projector,
                                                  transformer,
                                                  classifier,
                                                  criterion,
                                                  val_dl)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'projector_state_dict': projector.state_dict(),
                        'transformer_state_dict': transformer.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': min_val_loss,
                        }, ckpt_path/'best_model.pt')
            
        wandb.log({"epoch": epoch,
                   "val_acc/epoch": val_acc_avg.item(),
                   "val_loss/epoch": val_loss,
                   "val_acc_c0": val_acc[0].item(),
                   "val_acc_c1": val_acc[1].item(),
                   "val_acc_c2": val_acc[2].item()})
        
        print((f'Epoch: {epoch}, Train loss: {train_loss:.3f}, acc_avg: {train_acc_avg}, \n'
               f'Val loss: {val_loss:.3f}, acc_avg: {val_acc_avg}, acc (per class): {val_acc} \n'
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

    acc_metric_avg = Accuracy(task='multiclass',
                              num_classes=3,
                              average='macro').to(DEVICE)
    losses = 0
    n_frames = 0
    for samples in tqdm(dataloader, total=len(dataloader)):
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
        labels = torch.cat(labels).to(DEVICE)
        logits = classifier(encodings)
        if logits.shape[0] != labels.shape[0]:
            print('Logits and labels shape mismatch')
            print(f'Logits: {logits.shape}, labels: {labels.shape}')
            print(f'S1: {samples[0]["uuid"]}, S2: {samples[1]["uuid"]}')

        optimizer.zero_grad()

        loss = criterion(logits, labels)
        loss.backward()
        preds = nn.functional.softmax(logits, dim=1)
        acc_avg = acc_metric_avg(preds, labels)

        optimizer.step()
        losses += loss.item()
        n_frames += len(labels)
        wandb.log({"train_loss": loss.item(), "train_acc": acc_avg.item()})

    return losses / n_frames, acc_metric_avg.compute()


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
    preds = []
    ground_truth = []
    acc_metric_avg = Accuracy(task='multiclass',
                              num_classes=3,
                              average='macro').to(DEVICE)
    acc_metric = Accuracy(task='multiclass',
                          num_classes=3,
                          average=None).to(DEVICE)

    for samples in tqdm(dataloader, total=len(dataloader)):

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
        labels = torch.cat(labels).to(DEVICE)
        logits = classifier(encodings)

        loss = criterion(logits, labels)
        probs = nn.functional.softmax(logits, dim=1)
        acc_avg = acc_metric_avg(probs, labels)
        acc = acc_metric(probs, labels)

        losses += loss.item()
        n_frames += len(labels)
        wandb.log({"val_loss": loss.item(),
                   "val_acc": acc_avg.item(),
                   "val_acc_c0": acc[0].item(),
                   "val_acc_c1": acc[1].item(),
                   "val_acc_c2": acc[2].item()})
        preds.append(probs.argmax(dim=1))
        ground_truth.append(labels)

    preds = torch.Tensor(torch.cat(preds)).to(torch.uint8)
    ground_truth = torch.Tensor(torch.cat(ground_truth)).to(torch.uint8)
    wandb.log({"conf_matrix/val": wandb.plot.confusion_matrix( 
        preds=preds, y_true=ground_truth,
        class_names=['bckg', 'optimal', 'suboptimal'])})
   
    return losses / n_frames, acc_metric_avg.compute(), acc_metric.compute()


if __name__ == '__main__':
    wandb.login()
    main()
    wandb.finish()
