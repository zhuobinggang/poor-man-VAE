import mnist as M
import torch
t = torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn


def fill_map(the_map, key, val):
  if key not in the_map.keys():
    the_map[key] = []
  the_map[key].append(val)

# return {key: (?, 784)}
def get_data_map(ds = M.dataset_valid):
  data_map = {}
  for inpt,label in ds:
    fill_map(data_map, label, inpt)
  for key in data_map.keys():
    data_map[key] = t.stack(data_map[key])
  return data_map

def to_2d(ds = M.dataset_valid):
  data_map = {}
  all_inpts = []
  for inpt,label in ds:
    fill_map(data_map, label, inpt)
    all_inpts.append(inpt)
  all_inpts = t.stack(all_inpts)
  pca = PCA(n_components=2)
  pca.fit(all_inpts)
  for key in data_map.keys():
    data_map[key] = pca.transform(t.stack(data_map[key]))
  return data_map

# data_map: {key: (?, 2)}
def plot(data_map):
  plt.clf()
  cs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'gold', 'aqua']
  for idx, key in enumerate(data_map.keys()):
    datas_2d = data_map[key] # (?, 784)
    xs = [x for x, y in datas_2d]
    ys = [y for x, y in datas_2d]
    plt.scatter(xs, ys, c=cs[idx], label=key)
  plt.legend()
  plt.savefig('yes.png')
  return plt

# vae: trained vae
# dm: get_data_map(dataset)
# return: {key: (?, 2)}
def to_2d_by_data_map_and_vae(vae, dm_org):
  dm = {}
  all_z = []
  for key in dm_org.keys():
    dm[key] = vae._encoder(dm_org[key]) # (?, z_dim)
    all_z.append(dm[key])
  all_z = t.cat(all_z).detach().numpy() # (?, z_dim)
  pca = PCA(n_components=2)
  pca.fit(all_z)
  for key in dm.keys():
    dm[key] = pca.transform(dm[key].detach().numpy())
  return dm

def to_2d_by_data_map_and_vae_no_pca(vae, dm_org):
  dm = {}
  for key in dm_org.keys():
    dm[key] = vae._encoder(dm_org[key]).detach().numpy() # (?, z_dim)
  return dm


# dm: {key: (?, 784)}
# key: [0 -> 9]
def build_relative_pairs(dm):
  rp = []
  rp.append((random.choice(dm[0]), random.choice(dm[1]), random.choice(dm[2])))
  rp.append((random.choice(dm[1]), random.choice(dm[2]), random.choice(dm[3])))
  rp.append((random.choice(dm[2]), random.choice(dm[3]), random.choice(dm[4])))
  rp.append((random.choice(dm[3]), random.choice(dm[4]), random.choice(dm[5])))
  rp.append((random.choice(dm[4]), random.choice(dm[5]), random.choice(dm[6])))
  rp.append((random.choice(dm[5]), random.choice(dm[6]), random.choice(dm[7])))
  rp.append((random.choice(dm[6]), random.choice(dm[7]), random.choice(dm[8])))
  rp.append((random.choice(dm[7]), random.choice(dm[8]), random.choice(dm[9])))
  return rp
  
class Logic_VAE(nn.Module):
  # relative_pairs: [(zero_sample_input: (784), one_sample_input), (one_sample_input, two_sample_input), ...]
  def __init__(self, z_dim, relative_pairs):
    super().__init__()
    self.dense_enc1 = nn.Linear(28*28, 200)
    self.dense_enc2 = nn.Linear(200, 200)
    self.z_layer = nn.Linear(200, z_dim)
    self.dense_dec1 = nn.Linear(z_dim, 200)
    self.dense_dec2 = nn.Linear(200, 200)
    self.dense_dec3 = nn.Linear(200, 28*28)
    self.MSE = nn.MSELoss(reduction='sum')
    self.relative_pairs = relative_pairs

  def _encoder(self, x):
    x = F.relu(self.dense_enc1(x))
    x = F.relu(self.dense_enc2(x))
    z = self.z_layer(x)
    z = self._sample_z(z)
    return z
  
  def _sample_z(self, z):
    # epsilon = t.randn(z.shape)
    # return z + 0.2 * epsilon
    return z

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
    KL = x.shape[0] * 0.5 * self.KL_loss()
    y = self._decoder(z)
    reconstruction = self.MSE(y, x)
    # return KL + reconstruction
    return reconstruction

  def KL_loss(self):
    # TODO: sample from relative pairs, and minimize distance
    left, middle, right = random.choice(self.relative_pairs)
    z_left = self._encoder(left)
    z_middle = self._encoder(middle)
    z_right = self._encoder(right)
    # minimize distance
    return self.MSE(z_left, z_middle) + self.MSE(z_middle, z_right) - self.MSE(z_left, z_right)


def plot_by_vae(vae, dm):
  return plot(to_2d_by_data_map_and_vae(vae, dm))

def plot_by_vae_no_pac(vae, dm):
  return plot(to_2d_by_data_map_and_vae_no_pca(vae, dm))


class Logic_VAE_Contrast(Logic_VAE):
  def loss(self, x):
    z = self._encoder(x)
    y = self._decoder(z)
    reconstruction = self.MSE(y, x)
    return reconstruction


class Logic_VAE_Push_Other(Logic_VAE):
  def loss(self, x):
    z = self._encoder(x)
    KL = 10 * self.KL_loss()
    y = self._decoder(z)
    reconstruction = self.MSE(y, x)
    return KL + reconstruction

  def KL_loss(self):
    # TODO: sample from relative pairs, and minimize distance
    inpt_left, inpt_right = random.choice(self.relative_pairs)
    z_left = self._encoder(inpt_left)
    z_right = self._encoder(inpt_right)
    # minimize distance
    return - self.MSE(z_left, z_right)
