import torch
import torch.nn.functional as F 
from torch import nn


import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader


# Configuration
DEVICE = torch.device('cuda' is torch.cuda.is_avaliable() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4


# Model
class VariationalAutoEncoders(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim):
        super().__init__()

        #encoder
        self.img2_hid = nn.Linear(input_dim, hid_dim)
        self.hid2_hid = nn.Linear(hid_dim, hid_dim)
        self.hid2_mu = nn.Linear(hid_dim, z_dim)
        self.hid2_sigma = nn.Linear(hid_dim, z_dim)


        #decoder
        self.z2_hid = nn.Linear(z_dim, hid_dim)
        self.hid2_hidD = nn.Linear(hid_dim, hid_dim)
        self.hid2_img = nn.Linear(hid_dim, input_dim)

        self.relu = nn.ReLU()

    
    def encode(self, x):
        out1 = self.img2_hid(x)
        h = self.relu(self.hid2_hid(out1))
        mu, sigma = self.hid2_mu(h), self.hid2_sigma(h)
        return mu, sigma
    
    def decoder(self, z):
        out1 = self.z2_hid(z)
        h = self.relu(self.hid2_hidD(out1))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterised = mu + sigma * epslion
        x_recon = self.decode(z_reparameterised)

        return x_recon, mu, sigma



# Dataset loading
dataset = datasets.MNIST(root="dataset/", train=True, transfrom=transfromers.toTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffule=True)
model = VariantionalAutoEncoders(INPUT_SISM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:

        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, my, sigma = model(x)

        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -(torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)))

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model = model.to('cpu')
def inference(digit, num_ex = 1):

