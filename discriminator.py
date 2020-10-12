import torch
import torch.nn as nn
import config
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, gpus, classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(classes, classes)
        self.gpus = gpus
        self.net = nn.Sequential(
            nn.BatchNorm1d(config.image_size*config.image_size+classes),
            nn.Linear(config.image_size*config.image_size+classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            # N x channels_img x 64 x 64
            #nn.Conv2d(channels_img + int(np.prod((config.channels_img, config.image_size, config.image_size))), features_d, kernel_size=4, stride=2, padding=1),
            #nn.Linear(config.image_size*config.image_size + classes, config.image_size*config.image_size),
            #nn.unflatten(1, (config.batch_size, config.channels_img, config.image_size, config.image_size)),
            #nn.Conv2d(channels_img , features_d, kernel_size=4, stride=2, padding=1),
            #nn.LeakyReLU(0.2),
            # N x features_d x 32 x 32
            #nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(features_d * 2),
            #nn.LeakyReLU(0.2),
            #nn.Conv2d(
                #features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1
            #),
            #nn.BatchNorm2d(features_d * 4),
            #nn.LeakyReLU(0.2),
            #nn.Conv2d(
                #features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1
            #),
            #nn.BatchNorm2d(features_d * 8),
            #nn.LeakyReLU(0.2),
            # N x features_d*8 x 4 x 4
            #nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # N x 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x, labels):

        dis_input = torch.cat((x.view(x.size(0), -1), self.label_embedding(labels)), -1)
        
        #dis_input = torch.cat((x, self.label_embedding(labels)), -1)
        #lin = nn.Linear(config.image_size*config.image_size + config.classes, config.image_size*config.image_size)
        #dis = lin(dis_input)
        #dis_input_reshape = dis.view(config.batch_size, config.channels_img, config.image_size, config.image_size)
        return self.net(dis_input)
