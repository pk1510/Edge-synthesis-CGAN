import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import (
    DataLoader,
    )  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import discriminator # Import our models we've defined (from DCGAN paper)
import generator
import config
from torch.autograd import Variable

def weights_initialize(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


rootDir = r"E:/Prem/cnn/"
workers = 0
labels=[]
my_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=config.channels_img),
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.ImageFolder(
    root=rootDir, transform=my_transforms
)

dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers = workers, drop_last=True)

device = torch.device("cuda" if (torch.cuda.is_available() and config.gpus > 0) else "cpu")

# Create discriminator and generator

netG = generator.Generator(config.channels_noise, config.channels_img, config.features_g,config.gpus, config.classes).to(device)
netG.apply(weights_initialize)

netD = discriminator.Discriminator(config.channels_img, config.features_d,config.gpus, config.classes).to(device)
netD.apply(weights_initialize)
# Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr=config.lrD, betas=(config.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config.lrG, betas=(config.beta1, 0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, config.channels_noise, 1, 1).to(device)
real_summary = SummaryWriter(r"E:\Prem\cnn\real")   #logdir
fake_summary = SummaryWriter(r"E:\Prem\cnn\fake")
step = 0
images_real = []
images_fake = []
loss_g = []
loss_d = []
print("Training")

for epoch in range(config.num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)

        batch_size = data.shape[0]
        #booled = torch.where(data == 1, data, data-data)

        #for i in range(0, batch_size):
            #if (torch.sum(booled[i,0], dim=1)==booled.size(2)).any():
                #labels.insert(i, 0)
            #else:
                #labels.insert(i, 1)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))

        netD.zero_grad()
        #real_cpu = data.to(device)
        label = (torch.ones(config.batch_size) * 0.9).to(device)
        Labels = Variable((torch.LongTensor(config.labels[batch_idx*config.batch_size:(batch_idx+1)*config.batch_size])))
        #Labels = Variable(torch.LongTensor(labels))

        output = netD(data, Labels).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        noise = torch.randn(config.batch_size, config.channels_noise, 1, 1).to(device)
        fake = netG(noise, Labels)
        label = (torch.ones(config.batch_size) * 0.1).to(device)

        output = netD(fake.detach(), Labels).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        ### Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(config.batch_size).to(device)
        output = netD(fake, Labels).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()
        D_x = output.mean().item()
        # Print losses ocassionally and print to tensorboard
        if ((batch_idx+1)% 4 == 0 ):
        #if batch_idx in range(config.num_batches):
            step += 1
            print(
                f"Epoch [{epoch}/{config.num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f} D(x): {D_x:.4f}"
            )

            with torch.no_grad():
                fake = netG(fixed_noise, Labels[32:96]).detach().cpu()
                img_grid_real = torchvision.utils.make_grid(data[20:60], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[20:60], normalize=True)
                images_real.append(img_grid_real)
                images_fake.append(img_grid_fake)
                real_summary.add_image(                                           #adding logfiles in tensor board
                    "Real Images", img_grid_real, global_step=step
                )
                fake_summary.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
        loss_g.append(lossG)
        loss_d.append(lossD)
        labels=[]
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(loss_g,label="G")
plt.plot(loss_d,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in images_fake]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
#HTML(ani.to_jshtml())
