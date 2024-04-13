import torch
import torch.nn as nn
import torch.nn.functional as F

class Twice_Conv(nn.Module):
    def __init__(self, in_c, out_c, dropout_prob=0.5):
        super(Twice_Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return F.dropout(self.conv(x), p=0)


class Down_Layer(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down_Layer, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Twice_Conv(in_c, out_c)
        )

    def forward(self, x):
        return F.dropout(self.down(x), p=0)

class Up_Layer(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up_Layer, self).__init__()

        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = Twice_Conv(in_c, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)     #размер - [размер батча, количество каналов, длина, высота]
        delta_x = x2.shape[2] - x1.shape[2]
        delta_y = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [delta_x // 2, delta_x - delta_x // 2,
                        delta_y // 2, delta_y - delta_y // 2]) # дополняем x1 до x2 чтобы затем объединить их
        return self.conv(torch.cat([x2, x1], dim=1))

