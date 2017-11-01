import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from transform import Colorize
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop, Normalize
from transform import Scale, ToLabel, ReLabel
from gait import FCN
from datasets import CSTestSet
from PIL import Image
import numpy as np
import time
import cv2
import os
import shutil
# os.makedirs("./test")


image_transform = Compose([Scale((512, 256), Image.BILINEAR), 
                           ToTensor(),
                           Normalize([.485, .456, .406], [.229, .224, .225]),])

target_transform = Compose([
    Scale((512, 256), Image.NEAREST),
    ToLabel(),
    ReLabel(),
])

batch_size = 1
dst = CSTestSet("/root/group-incubation-bj", img_transform=image_transform, label_transform=target_transform)

testloader = data.DataLoader(dst, batch_size=batch_size,
                             num_workers=8)

model = torch.nn.DataParallel(FCN(6))
model.cuda()
model.load_state_dict(torch.load("./pth/seg-gait-4.pth"))
model.eval()

avr_time = 0.0

for j, data in enumerate(testloader):
    imgs, labels, names, original_size = data
    width = 512
    height = 256

    imgs = Variable(imgs.cuda())
    start_time = time.time()
    outputs_4, outputs_2, outputs = model(imgs)
    end_time = time.time()
    if j > 9:
        avr_time += (end_time - start_time)
    if j == 109:
        print avr_time/100.0
        exit()
    # 22 256 256
    filename = list(names)[0]
    for i, (output_4, output_2, output) in enumerate(zip(outputs_4, outputs_2, outputs)):
        output_4 = output_4.data.max(0)[1]

        output_4 = Colorize()(output_4)
        output_4 = np.transpose(output_4.numpy(), (1, 2, 0))
        img_4 = Image.fromarray(output_4, "RGB")
 
        output_2 = output_2.data.max(0)[1]

        output_2 = Colorize()(output_2)
        output_2 = np.transpose(output_2.numpy(), (1, 2, 0))
        img_2 = Image.fromarray(output_2, "RGB")

        output = output.data.max(0)[1]

        output = Colorize()(output)
        output = np.transpose(output.numpy(), (1, 2, 0))
        img = Image.fromarray(output, "RGB")
        if i == 0:
            img_4 = img_4.resize((width, height), Image.NEAREST)
            img_2 = img_2.resize((width, height), Image.NEAREST)
            img = img.resize((width, height), Image.NEAREST)
            img_4.save("test/" + filename.replace('.', "_4."))
            img_2.save("test/" + filename.replace('.', "_2."))
            img.save("test/" + filename)

