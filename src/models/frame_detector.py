import torch
import torch.nn as nn
from torch import Tensor
import math
from src.models.fuvai import YNet
from torch.nn.utils.rnn import pad_sequence
from typing import List
from pathlib import Path
import os

repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size = x.shape[0]

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


class YNetEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super(YNetEncoder, self).__init__()
        
        self.down_conv1 = pretrained_model.down_conv1
        self.down_conv2 = pretrained_model.down_conv2
        self.down_conv3 = pretrained_model.down_conv3
        self.down_conv4 = pretrained_model.down_conv4
        self.max_pool_2x2 = pretrained_model.max_pool_2x2
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = self.down_conv4[3].out_channels
        
    def forward(self, x):
        # x shape: (batch_size, 1, 1, 256, 256)
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.down_conv1(item)  
            x2 = self.down_conv2(self.max_pool_2x2(x1))  
            x3 = self.down_conv3(self.max_pool_2x2(x2))  
            x4 = self.down_conv4(self.max_pool_2x2(x3))  
            features = self.max_pool_2x2(x4)  
            data.append(features.unsqueeze(0))
        data = torch.cat(data, dim=0)
        data = self.avgpool(data)
        data = torch.flatten(data, -3)
        return data.squeeze()


def create_mask(src, pad_value=0):
    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
    src_padding_mask = torch.any(src == pad_value, dim=(2))
    #(src == padding_value)
   
    return src_mask, src_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000,
                 postype: str = 'learnable'):
        super(PositionalEncoding, self).__init__()

        if postype == 'learnable':
            self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size))
        elif postype == 'sin_cos':
            den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
            pos = torch.arange(0, maxlen).reshape(maxlen, 1)
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
            pos_embedding = pos_embedding.unsqueeze(0)  # batch size first
            self.register_buffer('pos_embedding', pos_embedding)
        else:
            raise NameError(f'Param type {postype} is not implemented.')

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    

class Transformer(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 nheads=6,
                 num_encoder_layers=6,
                 maxlenght=840,
                 pos_embed: str = 'learnable',
                 padding_value=0):
        
        super().__init__()
        self.padding_value = padding_value
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=nheads,
                                                   dim_feedforward=2048,
                                                   dropout=0.1,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.model = nn.TransformerEncoder(encoder_layer,
                                           num_encoder_layers,
                                           encoder_norm)
        
        self.pos_embed = PositionalEncoding(emb_size=hidden_dim,
                                            dropout=0.1,
                                            maxlen=maxlenght,
                                            postype=pos_embed)
        
    def forward(self, x: List[Tensor]):
        # (batch, seq, feature)
        x = pad_sequence(x, batch_first=True,
                         padding_value=self.padding_value)
        # make masks
        src_mask, src_padding_mask = create_mask(x, self.padding_value)

        # add positional encoding
        x = self.pos_embed(x)       

        x = self.model(x, mask=src_mask,
                       src_key_padding_mask=src_padding_mask)
        
        return x, src_padding_mask


class DETRdemo(nn.Module):
    def __init__(self,
                 pretrained_encoder,
                 num_classes=3,
                 hidden_dim=512,
                 nheads=8,
                 num_encoder_layers=6,
                 maxlenght=840):
        super().__init__()

        # ckpt_path = data_path / 'fuvai_weights.pt'
        # model = YNet(1, 64, 1)
        # ckpt = torch.load(ckpt_path)
        # model.load_state_dict(ckpt)
        self.backbone = pretrained_encoder  # YNetEncoder(YNet())

        self.transformer = Transformer()

        self.proj = nn.Linear(self.backbone.out_channels,
                              hidden_dim)

        self.linear_class = SimpleDenseNet(hidden_dim,
                                           256,
                                           256,
                                           256,
                                           num_classes)


def build_models(hidden_dim=768):

    # load pretrained model and freeze
    ckpt_path = repo_path / 'data' / 'fuvai_weights.pt'
    pretrained_model = YNet(1, 64, 1)
    ckpt = torch.load(ckpt_path)
    pretrained_model.load_state_dict(ckpt)

    encoder = YNetEncoder(pretrained_model=pretrained_model)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # projection layer
    proj = nn.Linear(encoder.out_channels,
                     hidden_dim)

    # transformer
    transformer = Transformer(hidden_dim=hidden_dim)

    # classifier
    classifier = SimpleDenseNet(input_size=hidden_dim,
                                output_size=3)

    return encoder, proj, transformer, classifier
