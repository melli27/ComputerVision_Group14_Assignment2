# TUWIEN - CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE 14

from typing import List
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MaskClassifier(nn.Module):

    def __init__(self, name, img_size=64, dropout: float = 0, batch_norm: bool = False):
        """
        Initializes the network architecture by creating a simple CNN of convolutional and max pooling layers.
        
        Args:
        - name: The name of the classifier.
        - img_size: Size of the input images.
        - dropout (float): Dropout rate between 0 and 1.
        - batch_norm (bool): Determines if batch normalization should be applied.
        """
        super(MaskClassifier, self).__init__()
        self.name = name
        self.img_size = img_size
        self.batch_norm = batch_norm

        # student code start
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)

        # 64x64 -> 62x62 -> 31x31 -> 29x29 -> 14x14
        fc1_input_size = 32 * 14 * 14
        self.fc1 = nn.Linear(fc1_input_size, 1)
        # student code end


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the predefined layers of the network to x.
        
        Args:
        - x (Tensor): Input tensor to be classified [batch_size x channels x height x width].
        
        Returns:
        - Tensor: Output tensor after passing through the network layers.
        """
        
        # student code start
        # Apply layers here
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch

        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        # student code end

        return x