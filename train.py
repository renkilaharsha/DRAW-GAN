import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets,transforms
import torch.utils
from draw_model import DrawModel
from config import *
from utility import *
import torch.nn.utils
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
#from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
#from torchvision.utils import save_image
#from torchvision.datasets import MNIST
import os
sigmoid= nn.Sigmoid()
torch.set_default_tensor_type('torch.FloatTensor')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Discriminator(nn.Module):  
    def __init__(self,N,z_dim):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, 100)
        self.lin3 = nn.Linear(100, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x1 = F.relu(x)
        return F.sigmoid(self.lin3(x1)) ,x1       
 
discriminator = Discriminator(500,1024)

       
def disc_loss(x,x_hat):
    EPS = 1e-15
    #z_real_gauss = Variable(torch.randn(bs,784)*5)
    z_real = discriminator(x)
            
    z_fake = discriminator(x_hat)
    d_loss = -torch.mean(torch.log(z_real +EPS )+ torch.log(1-z_fake + EPS))
    g_loss = -torch.mean(torch.log(z_fake + EPS))

    return g_loss,d_loss

#FData Preparation
img_transform = transforms.Compose([transforms.ToTensor()])
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, -1])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][:-1]).reshape(32, 32).astype('uint16')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

img_transform = transforms.Compose([transforms.ToTensor()])
dataset=CustomDatasetFromCSV('/home/17mcmi06/harsha/project/draw_tel_unbalanced/UHTelPCC.csv',32,32, transform=img_transform)

train_loader = DataLoader(dataset,batch_size=batch_size,shuffle =True)

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))
optimizer_loss = optim.Adam(model.parameters(),lr=0.00005,betas=(beta1,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005)
EPS =1e-15


if USE_CUDA:
    model.cuda()
    discriminator.cuda()

def generate_image(count):
    x,_ = model.generate(64)
    return x

def save_example_image():
    train_iter = iter(train_loader)
    #print(train_iter.shape())
    data, _ = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 32, 32)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    save_example_image()
    data_ix = np.empty((1,785))
  #  train()

    avg_loss = 0
    count = 0
    for epoch in range(epoch_num):
        for data,lables in train_loader:
            bs = data.size()[0]
            y_real = Variable(torch.ones(bs,1), requires_grad=False)
            y_fake = Variable(torch.zeros(bs,1), requires_grad=False)


            data = Variable(data.float()).view(bs, -1).cuda()
            
            
           
            loss,x_f,recon = model.loss(data)
            
            _,x_p = model.generate(bs)

            ld_r,fd_r= discriminator(data)
            ld_f,fd_f= discriminator(x_f)
           # print(x_p[-1].shape)
            ld_p,fd_p= discriminator(x_p[-1])
 
            #loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
            loss_D = torch.mean(torch.log(ld_r+EPS) +torch.log(1-ld_f+EPS) +torch.log(1-ld_p+EPS))

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph = True)
            optimizer_D.step()

    #------------E & G training--------------

    #loss corresponding to -log(D(G(z_p)))
            #loss_GD = F.binary_cross_entropy(ld_p, y_real)
            criterion = nn.BCELoss()
    #pixel wise matching loss and discriminator's feature matching loss
            #loss_G = 0.5 * (0.01*(x_f - data).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) 
            loss_G = criterion(x_f ,data)*A*B
            #loss_G = ( (0.01*criterion(x_f ,data) + criterion(fd_f ,fd_r.detach())) )*A*B
            #loss_G = criterion(fd_f ,fd_r.detach())*A*B 
            optimizer.zero_grad()
            (loss + 0.8*loss_G -loss_D).backward()

            
            
            
           # avg_loss += loss.cpu().data.numpy()
            
            #loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            
            count += 1
            #save_inputimage(data,count)
            #save_reconimage(recon,count)
            print ('Epoch-{}; Count-{}; loss:{}; DISC_loss:{}; loss_gd:{}; loss_g:{} ;'.format(epoch, count, loss ,loss_D,loss + 1.1*loss_G -loss_D ,loss_G))
            if count % 100 == 0:
                print ('Epoch-{}; Count-{}; loss:{}; DISC_loss:{}; loss_gd:{}; loss_g:{} ;'.format(epoch, count, loss ,loss_D,loss + 1.1*loss_G -loss_D,loss_G))
                torch.save(model.state_dict(),'save/weights_%d.tar'%(count))
                #save_inputimage(data.cpu(),count)
                #save_reconimage(recon.cpu().data.numpy(),count)
                x=generate_image(count)
                save_image(x,count)
                avg_loss = 0
    torch.save(model.state_dict(), 'save/weights_final.tar')
    generate_image(count)

