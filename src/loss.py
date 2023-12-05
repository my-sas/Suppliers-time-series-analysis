import torch
import torch.nn as nn


class BidirectionalMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, input_, target):
        return super().forward(
            input_,
            torch.stack((target, target.flip(dims=[1])))
        )
