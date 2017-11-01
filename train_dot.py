#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from dot import FCN 
from datasets import CSDataSet
from loss import CrossEntropy2d, CrossEntropyLoss2d
from transform import ReLabel, ToLabel, ToSP, Scale, Augment
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from PIL import Image
import numpy as np

import utils
from image_augmentor import ImageAugmentor

image_augmentor = ImageAugmentor()

NUM_CLASSES = 6
MODEL_NAME = "seg-dot"

input_transform = Compose([
    Scale((512, 256), Image.BILINEAR),
    Augment(0, image_augmentor),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
target_transform = Compose([
    Scale((512, 256), Image.NEAREST),
    ToLabel(),
    ReLabel(),
])

target_2_transform = Compose([
    Scale((256, 128), Image.NEAREST),
    ToLabel(),
    ReLabel(),
])

target_4_transform = Compose([
    Scale((128, 64), Image.NEAREST),
    ToLabel(),
    ReLabel(),
])

trainloader = data.DataLoader(CSDataSet("/root/group-incubation-bj", split="train",
                                        img_transform=input_transform, label_transform=target_transform,
                                        label_2_transform=target_2_transform, label_4_transform=target_4_transform),
                                        batch_size=10, shuffle=True, pin_memory=True)

valloader = data.DataLoader(CSDataSet("/root/group-incubation-bj", split="val",
                                      img_transform=input_transform, label_transform=target_transform,
                                      label_2_transform=target_2_transform, label_4_transform=target_4_transform),
                                      batch_size=1, pin_memory=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(FCN(NUM_CLASSES))
    model.cuda()

epoches = 8
lr = 1e-3

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# pretrained_dict = torch.load("./pth/fcn-deconv-40.pth")
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# model.load_state_dict(torch.load("./pth/seg-skip-all-25.pth"))

model.train()

x_index = 1

for epoch in range(epoches):
    running_loss = 0.0
    iter_loss = 0.0
    for i, (images, labels, labels_2, labels_4) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            labels_2 = Variable(labels_2.cuda())
            labels_4 = Variable(labels_4.cuda())
        else:
            images = Variable(image)
            labels = Variable(labels)
            labels_2 = Variable(labels_2)
            labels_4 = Variable(labels_4)

        optimizer.zero_grad()
        outputs_4, outputs_2, outputs = model(images)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(outputs_2, labels_2)
        loss3 = criterion(outputs_4, labels_4)

        if epoch + 1 < 3:
            loss = loss3
        else:
            loss = (loss1 + loss2 + loss3)/3.0

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        iter_loss += loss.data[0]
        if (i + 1) % 100 == 0:
            print("Iter [%d] Loss: %.4f" % (i+1, iter_loss/100.0))
            iter_loss = 0.0

        if (i + 1) % 300 == 0:
            utils.plot(MODEL_NAME + "-train_loss", x_index, running_loss/300.0)
            print("Epoch [%d] Loss: %.4f" % (x_index, running_loss/300.0))
            running_loss = 0

            val_loss = 0.0
            for j, (images, labels, labels_2, labels_4) in enumerate(valloader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                    labels_2 = Variable(labels_2.cuda())
                    labels_4 = Variable(labels_4.cuda())
                else:
                    images = Variable(image)
                    labels = Variable(labels)
                    labels_2 = Variable(labels_2)
                    labels_4 = Variable(labels_4)
        
                outputs_4, outputs_2, outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(outputs_2, labels_2)
                loss3 = criterion(outputs_4, labels_4)

                loss = (loss1 + loss2 + loss3) / 3.0
                val_loss += loss.data[0]
        
            print("Val [%d] Loss: %.4f" % (x_index, val_loss/len(valloader)))
            utils.plot(MODEL_NAME + "-val_loss", x_index, val_loss/len(valloader))
            x_index += 1
            val_loss = 0

    if (epoch+1) % 1 == 0:
        if (epoch + 1) % 3 == 0:
            lr /= 10.0
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

        torch.save(model.state_dict(), "./pth/" + MODEL_NAME + ("-%d.pth" % (epoch+1)))


torch.save(model.state_dict(), "./pth/" + MODEL_NAME + ".pth")
