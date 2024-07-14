import torch
import torch.nn as nn
from src.models.fuvai import YNet

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
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
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
        batch_size, channels, width, height = x.size()

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

class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.backbone = YNetEncoder(YNet())
        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = SimpleDenseNet(hidden_dim,
                                           256, 256, 256, num_classes)
