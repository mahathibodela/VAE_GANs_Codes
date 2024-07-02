import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvison.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transfroms as transforms
from torch.utils.tensorboard import SummaryWriter


# Model
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyRelu(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyRelu(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.gan(x)


# Hyperparamters etc..
device = "cuda" if torch.cuda.is_avaliable() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epoches = 25

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, zdim)).to(deivce) # ye kyon.. and pehle bhi for each digit VAE ko train karna chahiye tha na
transforms = transforms.Compose(
    [transfroms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
)

dataset = datasets.MNIST(root="dataset/", train=True, transfrom=transfromers, download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffule=True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAn_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAn_MNIST/real")


for epoch in range(num_epoches):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.size(-1, 784).to(device)
        batch_size = real.shape[0]
        
        # train discrimnator
        fake = gen(fixed_noise) # ye noise firse kuch karna padegak ya
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach(h)).view(-1)
        lossD_fake = criterion(disc_fake, torch.ones_like(disc_fake))
        lossD = (lossD_real + lossD_real) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()
        
        #train generator
        output = disc(fake).view(-1) # fake varialbes ka gradients honge kya
        lossG = criterion(output, torch.ones_like(ouput))
        gan.zero_grad()
        lossG.backward()
        opt_gen.step()
    
