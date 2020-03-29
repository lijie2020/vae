import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU()

        )#14*14
        self.conv2=nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )#3

        self.conv3=nn.Sequential(
            nn.Conv2d(256,512,3,2,0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(512,2,3,1,0)
        )
    def forward(self, x):
        y1=self.conv1(x)
        y2=self.conv2(y1)
        y3=self.conv3(y2)
        y4=self.conv4(y3)
        miu=y4[:,:1,:,:]
        sigma=y4[:,1:,:,:]
        return miu,sigma


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.ConvTranspose2d(128,512,3,1,0),
            nn.BatchNorm2d(512),
            nn.ReLU()

        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512,256, 3, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256,128, 3, 2, 1,1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 3, 2, 1,1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, miu,log_sigma,z):
        x=z*torch.exp(log_sigma)+miu
        x=x.permute([0,3,1,2])
        y1=self.conv1(x)
        y2=self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        return y4




class Net_total(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
    def forward(self, x,z):
        miu,log_sigma=self.encoder.forward(x)
        output=self.decoder.forward(miu,log_sigma,z)

        return miu,log_sigma,output

