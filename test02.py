import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import test01
from torchvision import transforms,datasets

if __name__ == '__main__':
    tf=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])]
    )

    traindata=datasets.MNIST(".\data\MNIST",train=True,transform=tf,download=False)
    trainloader=DataLoader(traindata,100,shuffle=True)

    net=test01.Net_total()
    opt=torch.optim.Adam(net.parameters())
    loss_fn=nn.MSELoss(reduction="sum")


    for epoch in  range(5):
        for i ,(img,lable) in enumerate(trainloader):
            z=torch.randn(128)
            miu,log_sigma,out_img=net(img,z)

            en_loss=torch.mean((-torch.log(log_sigma**2)+miu**2+log_sigma**3-1)*0.5)
            de_loss=loss_fn(out_img,img)
            loss=en_loss+de_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%10==0:
                print(loss.item())
                fake_img=out_img.data#假图片
                img=img.data

                save_image(fake_img,"./data/img/{}-fake_img.png".format(epoch+i),nrow=10)
                save_image(img, "./data/img/{}-real_img.png".format(epoch + i), nrow=10)
