import torch
from torchvision import models
from torch import nn
from utils.metrics import captcha_len, num_classes


class CaptchaNet(nn.Module):
    def __init__(self):
        super(CaptchaNet, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.out_features = 512 # 2048
        self.out_h = 2
        self.out_w = 7
        self.captcha_len = captcha_len
        self.num_classes = num_classes

        self.dropout0 = nn.Dropout2d(p=0.4)
        self.linear1 = nn.Linear(self.out_features, self.captcha_len)
        self.linear2 = nn.Linear(self.out_h*self.out_w, self.num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) # [batch_size, out_features, out_h, out_w]

        x = self.dropout0(x)
        x = x.reshape(-1, self.out_features, self.out_h*self.out_w) # [batch_size, out_features, out_h*out_w]
        x = x.transpose(2, 1) # [batch_size, out_h*out_w, out_features]
        x = self.linear1(x) # [batch_size, out_h*out_w, captcha_len]
        x = x.transpose(2, 1) # [batch_size, captcha_len, out_h*out_w]
        x = self.linear2(x) # [batch_size, captcha_len, num_classes]

        return x


if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 50, 200)
    out = CaptchaNet()(input_tensor)
    print(out.shape)