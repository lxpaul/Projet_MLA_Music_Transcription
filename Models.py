import torch as th
import torch.nn as nn


class BranchYp(nn.Module):
    def __init__(self):
        super(BranchYp, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,5), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3,39), padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(5,5), padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
class BranchYn(nn.Module):
    def __init__(self):
        super(BranchYn, self).__init__()
        self.layers = nn.Sequential5(
            nn.Conv2d(1, 32, kernel_size=(7,7), stride=(1,3)),
            nn.ReLU(),
            nn.Conv2d(7, 3, kernel_size=(7,3)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
class BranchY0(nn.Module):
    def __init__(self):
        super(BranchY0, self).__init__()
        self.layer_1 = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,3))
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.acti_1 = nn.ReLU()
        self.layer_2 = nn.Conv2d(33, 1, kernel_size=(3,3))
        self.sigmo = nn.Sigmoid()

    def forward(self,x,Yn):
        x = self.layer_1(x)
        x = self.batch_norm_1(x)
        x = self.acti_1(x)
        x = th.concatenate((x,Yn))
        x = self.layer_2(x)
        return self.sigmo(x)