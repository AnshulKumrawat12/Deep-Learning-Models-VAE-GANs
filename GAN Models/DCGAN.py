import enum
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard  import SummaryWriter


#Model Definition
### Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        self.Disc = nn.Sequential(
            # Input : N x channels_img x 64 x 64
            nn.Conv2d( channels_img, features_d, kernel_size = 4, stride = 2, padding=1), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 16 x 16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 8 x 8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size = 4,stride=2, padding=0), # 1 x 1
            nn.Sigmoid()

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.Disc(x)


### Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()

        self.Gen = nn.Sequential(
            #Input : N x z_dim x 1 x 1
            self._block(z_dim, features_g* 16, 4, 1, 0), # N x features_g*16 x 4 x 4      ---> suppose features_g = 64, Then we give features_g*16 = 1024 as given in paper
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32 x 32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size = 4, stride = 2, padding = 1), 
            nn.Tanh() # ------> [-1, 1]
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.Gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


## To test models are working well or not
# def test():
#     N, in_channels, H, W = 8, 3, 64, 64
#     z_dim = 100
#     x = torch.randn((N, in_channels, H, W))
#     disc = Discriminator(in_channels, 8)
#     initialize_weights(disc)
#     assert disc(x).shape == (N,1,1,1)
#     gen = Generator(z_dim, in_channels, 8)
#     z = torch.randn((N, z_dim, 1,1))
#     assert gen(z).shape == (N, in_channels, H, W)
#     print("success")

# test()



# Hyper Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 5
features_disc =  64
features_gen = 64


transforms = transforms.Compose(
    [transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5 for _ in range(channels_img)], std = [0.5 for _ in range(channels_img)])]
)


#Dataset Loading
dataset = datasets.MNIST(root= "dataset/", transform= transforms, train = True, download= True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Model Calling and Optimizer and Loss define
gen = Generator(z_dim, channels_img, features_gen)
disc = Discriminator(channels_img, features_disc)
initialize_weights(gen)
initialize_weights(disc)

opt_disc = optim.Adam(disc.parameters(), lr = learning_rate)
opt_gen = optim.Adam(gen.parameters(), lr = learning_rate)
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/DCGAN_MNIST/real")
writer_fake = SummaryWriter(f"runs/DCGAN_MNIST/fake")
step = 0


##Training Loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        #Define input for Generator and Discriminator
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1))
        fake = gen(noise).to(device)

        ## Train Discriminator max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (loss_disc_fake + loss_disc_real) / 2

        disc.zero_grad()
        disc_loss.backward(retain_graph = True)
        opt_disc.step()


        ##Train Generator min log( 1 - D(G(z))) ---------> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}/{num_epochs} , D_loss: {disc_loss:.4f}, G_loss: {loss_gen:.4f}")


            with torch.no_grad():
                fake = gen(fixed_noise).to(device)
                # Take out (upto) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step +=1


#%%
import matplotlib.pyplot as plt

plt.imshow(img_grid_real.permute(1, 2, 0).cpu())
plt.imshow(img_grid_fake.permute(1, 2, 0).cpu())