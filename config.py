import torch
lrD = 0.0001
lrG = 0.0001
batch_size = 250
num_batches = int(1000/batch_size)
image_size = 8
channels_img = 1
channels_noise = 100
num_epochs = 50
beta1 = 0.5
gpus = 0
# For how many channels Generator and Discriminator should use
features_d = 16
features_g = 16
classes = 2
labels=[]
for i in range(0,1000):
    if i < 500:
        labels.append(0)
    else:
        labels.append(1)
