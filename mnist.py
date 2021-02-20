import torch
t = torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)


class VAE(nn.Module):
    def __init__(self, z_dim):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(28*28, 200)
      self.dense_enc2 = nn.Linear(200, 200)
      self.dense_encmean = nn.Linear(200, z_dim)
      self.dense_encvar = nn.Linear(200, z_dim)
      self.dense_dec1 = nn.Linear(z_dim, 200)
      self.dense_dec2 = nn.Linear(200, 200)
      self.dense_dec3 = nn.Linear(200, 28*28)
    
    def _encoder(self, x):
      x = F.relu(self.dense_enc1(x))
      x = F.relu(self.dense_enc2(x))
      mean = self.dense_encmean(x)
      var = F.softplus(self.dense_encvar(x))
      return mean, var
    
    def _sample_z(self, mean, var):
      epsilon = torch.randn(mean.shape)
      return mean + torch.sqrt(var) * epsilon
 
    def _decoder(self, z):
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x):
      mean, var = self._encoder(x)
      z = self._sample_z(mean, var)
      x = self._decoder(z)
      return x, z
    
    def loss(self, x):
      mean, var = self._encoder(x)
      KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
      z = self._sample_z(mean, var)
      y = self._decoder(z)
      reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      lower_bound = [-KL, reconstruction]                                      
      return -sum(lower_bound)



class My_VAE(nn.Module):
    def __init__(self, z_dim):
      super().__init__()
      self.dense_enc1 = nn.Linear(28*28, 200)
      self.dense_enc2 = nn.Linear(200, 200)
      self.z_layer = nn.Linear(200, z_dim)
      self.dense_dec1 = nn.Linear(z_dim, 200)
      self.dense_dec2 = nn.Linear(200, 200)
      self.dense_dec3 = nn.Linear(200, 28*28)
      self.z_anchor = t.randn(z_dim)
      self.MSE = nn.MSELoss()
    
    def _encoder(self, x):
      x = F.relu(self.dense_enc1(x))
      x = F.relu(self.dense_enc2(x))
      z = self.z_layer(x)
      z = self._sample_z(z)
      return z
    
    def _sample_z(self, z):
      epsilon = torch.randn(z.shape)
      return z + epsilon
 
    def _decoder(self, z):
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x):
      z = self._encoder(x)
      z = self._sample_z(z)
      x = self._decoder(z)
      return x, z
    
    def loss(self, x):
      z = self._encoder(x)
      KL = self.MSE(z, self.z_anchor.repeat(x.shape[0], 1))
      y = self._decoder(z)
      reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      lower_bound = [-KL, reconstruction]                                      
      return -sum(lower_bound)

class My_VAE_V2(nn.Module):
    def __init__(self, z_dim):
      super().__init__()
      self.fw1 = nn.Linear(28*28, 28*14)
      self.z_layer = nn.Linear(28*14, z_dim)
      self.fw2 = nn.Linear(z_dim, 28*14)
      self.recover_layer = nn.Linear(28*14, 28*28)
      self.MSE = nn.MSELoss(reduction='sum')
    
    def _encoder(self, x):
      z = self.add_disturbance(self.z_layer(F.sigmoid(self.fw1(x))))
      return z
    
    def add_disturbance(self, z):
      epsilon = torch.randn(z.shape)
      return z + epsilon
 
    def _decoder(self, z):
      o = F.sigmoid(self.recover_layer(F.sigmoid(self.fw2(z))))
      return o

    def forward(self, x):
      z = self._encoder(x)
      o = self._decoder(z)
      return o, z
    
    def loss(self, x):
      z = self._encoder(x)
      KL = z.pow(2).sum().sqrt() # Force z decay to zero
      y = self._decoder(z)
      reconstruction = self.MSE(y, x)
      return KL + reconstruction
      # reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      # lower_bound = [-KL, reconstruction]                                      
      # return -sum(lower_bound)


def train(model, epoch = 20):
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  model.train()
  for i in range(epoch):
    losses = []
    for x, t in dataloader_train:
        model.zero_grad()
        y = model(x)
        loss = model.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
    print("EPOCH: {} loss: {}".format(i, np.average(losses)))
  return model


def test(model):
  fig = plt.figure(figsize=(10, 3))
  zs = []
  for x, t in dataloader_valid:
      # original
      for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      # generate from x
      y, z = model(x)
      zs.append(z)
      y = y.view(-1, 28, 28)
      for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      # generate from z
      dds = [z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)]
      y2 = [model._decoder(dd).detach().view(28,28).numpy() for dd in dds]
      for i, im in enumerate(y2):
        ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
        ax.imshow(im, 'gray')
      break
  plt.savefig('dd.png')
  return fig
