
#Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim):  #28x28x1 ----> 784, mnist dataset
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.Gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28x28x1 ---> 784
            nn.Tanh() # ----> to make sure that the values of pixels are in between -1 and 1, as in the input only we are normalizing the values between -1 and 1 hence our output should also be in same form
        )

    def forward(self, x):
        return self.Gen(x)



#%%

#Hyperparameters 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
batch_size = 32
z_dim = 64 # 128, 256
image_dim = 28 * 28 * 1 # 784
num_epochs = 50

#Model Calling
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5), std = (0.5))]
)

#Dataset Loading
dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0


#%%
for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.view(-1, 784).to(device)

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)

        # It changes BCE loss and takes y as 1 thus it calculates only first term in loss. 
        # In BCE loss -ve sign is also there and it is minimizing that loss.And Here we want to maximize it. So they both are equal, so no changes in it.
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) 

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph = True) ## If we want to apply multiple backward function inside one loop, then we have to make retain_graph = True for previous functions
        opt_disc.step()


        ###Train Generator min log(1 - D(G(z))) -------> max D(G(z))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx == 0:
            print(f" Epoch : [{epoch/num_epochs}] Loss D : {lossD: .4f}, Loss G: {lossG: .4f}")


        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1,28,28)
            real = real.reshape(-1,1,28,28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
            img_grid_real = torchvision.utils.make_grid(real, normalize = True)

            writer_fake.add_image("MNIST Fake Images ", img_grid_fake, global_step= step)
            writer_real.add_image("MNIST Real Images ", img_grid_real, global_step= step)
        
        step+=1



#%%
import matplotlib.pyplot as plt

plt.imshow(img_grid_real.permute(1, 2, 0).cpu())
plt.imshow(img_grid_fake.permute(1, 2, 0).cpu())