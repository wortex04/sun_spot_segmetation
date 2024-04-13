from main_model.u_net_blocks import Twice_Conv, Down_Layer, Up_Layer
import torch
import torch.nn as nn
import numpy as np
import cv2


class U_Net(nn.Module):

    def __init__(self):
        super(U_Net, self).__init__()

        self.conv = Twice_Conv(3, 64)

        self.conv_down1 = Down_Layer(64, 128)
        self.conv_down2 = Down_Layer(128, 256)
        self.conv_down3 = Down_Layer(256, 512)
        self.conv_down4 = Down_Layer(512, 1024)

        self.conv_up1 = Up_Layer(1024, 512)
        self.conv_up2 = Up_Layer(512, 256)
        self.conv_up3 = Up_Layer(256, 128)
        self.conv_up4 = Up_Layer(128, 64)

        self.last = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_down3(x3)
        x5 = self.conv_down4(x4)

        out = self.conv_up1(x5, x4)
        out = self.conv_up2(out, x3)
        out = self.conv_up3(out, x2)
        out = self.conv_up4(out, x1)

        return self.last(out)

    def masked_image(self, mask, img):  # mask: [600, 600], values: 0 or 1, image: [3, 600, 600]
        binary_mask = (mask.copy()[:, :] != 1.).astype(np.uint8) * 255  # if mask's value is 0 it's was transform to 255
        result_image = img.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            result_image = cv2.rectangle((result_image), (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result_image

    def infer(self, x, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(x, torch.Tensor):
            x = x.to('cpu').numpy()
            if x.shape[-1] != 3:
                x = x.transpose(2, 0, 1)

        if len(x.shape) == 4:
            x = x[0]
        img = x.copy()  #image.shape = [600, 600, 3]
        for i in range(600):
            for j in range(600):
                if (300 - i) ** 2 + (300 - j) ** 2 > 290 ** 2:
                    x[i][j] = 126

        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        x = np.stack([blurred, blurred, blurred])
        x = torch.from_numpy(x).unsqueeze(0)
        x = x / 255.
        x = x.to(device)

        x1 = self.conv(x)
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_down3(x3)
        x5 = self.conv_down4(x4)

        out = self.conv_up1(x5, x4)
        out = self.conv_up2(out, x3)
        out = self.conv_up3(out, x2)
        out = self.conv_up4(out, x1)
        out = self.last(out)
        out = out.squeeze(0)  # [1, 2, 600, 600]  -> [2, 600, 600]
        mask = out.argmax(dim=0)  # mask.shape = [600, 600]
        mask = mask.detach().to('cpu').numpy()
        mask_new = np.zeros((600, 600))
        for i in range(600):
            for j in range(600):
                if mask[i][j] == 0:
                    mask_new[i][j] = 1
        return self.masked_image(mask_new, img), mask, out, x
