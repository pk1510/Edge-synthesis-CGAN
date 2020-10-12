import torch
import torch.nn as nn
import config
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g,gpus, classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(classes, classes)
        self.gpus = gpus
        self.net = nn.Sequential(
            nn.BatchNorm1d(config.channels_noise+classes),
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,64),
            # N x channels_noise x 1 x 1
            #nn.ConvTranspose2d(
                #channels_noise, features_g * 16, kernel_size=4, stride=1, padding=0
            #),
            #nn.BatchNorm2d(features_g * 16),
            #nn.ReLU(),
            # N x features_g*16 x 4 x 4
            #nn.ConvTranspose2d(
                #features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1
            #),
            #nn.BatchNorm2d(features_g * 8),
            #nn.ReLU(),
            #nn.ConvTranspose2d(
                #features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1
            #),
            #nn.BatchNorm2d(features_g * 4),
            #nn.ReLU(),
            #nn.ConvTranspose2d(
                #features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1
            #),
            #nn.BatchNorm2d(features_g * 2),
            #nn.ReLU(),
            #nn.ConvTranspose2d(
                #features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            #),
            # N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def forward(self, x, labels):
        gen_input = torch.cat((self.label_emb(labels), x.view(x.size(0), -1)), -1)
        #lin = nn.Linear(config.channels_noise+config.classes, config.channels_noise)
        #gen = lin(gen_input)
        #trained = gen.view(x.size(0), config.channels_noise, 1, 1)
        train = self.net(gen_input)
        trained = train.view(train.size(0), 1, 8, 8)
        return trained
