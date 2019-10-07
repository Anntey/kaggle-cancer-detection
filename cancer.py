
#############
# Libraries #
#############

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Linear, CrossEntropyLoss
from torch.nn.functional import leaky_relu
from torch.optim import Adamax
from torchvision.transforms import Compose, ToPILImage, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

#############
# Load data #
#############

train_df = pd.read_csv("./input/train_labels.csv")

train, val = train_test_split(
        train_df,
        stratify = train_df["label"],
        shuffle = True,
        test_size = 0.1
)

##################
# Visualize data #
##################
    
random_img_ids = np.random.choice(train_df["id"], 18)

fig = plt.figure(figsize = (11, 5))
for i, img_id in enumerate(random_img_ids):
    plt.subplot(3, 6, i + 1)
    img = cv2.imread("./input/train/" + img_id + ".tif")
    label = train_df["label"].loc[train_df["id"] == img_id].values[0]
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis("off")
plt.tight_layout() 
   
##################
# Data generator #
##################

class CancerDataset(Dataset):
    
    def __init__(self, df_data, path, augmentations = None):
        self.df = df_data.values
        self.path = path
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_id, label = self.df[index]
        img = cv2.imread(self.path + img_id + ".tif")
        
        if self.augmentations is not None:
            img = self.augmentations(img)
            
        return img, label

augs_train = Compose([
        ToPILImage(),
        Pad(64, padding_mode = "reflect"),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(20),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augs_val = Compose([
        ToPILImage(),
        Pad(64, padding_mode = "reflect"),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_train = CancerDataset(train, "./input/train/", augmentations = augs_train)
dataset_val = CancerDataset(val, "./input/train/",  augmentations = augs_val)

batch_size = 128

train_gen = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
val_gen = DataLoader(dataset_val, batch_size = batch_size // 2, shuffle = False)

#################
# Specify model #
#################

class SimpleCNN(nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(3, 32, 3, padding = 2)
        self.conv2 = Conv2d(32, 64, 3, padding = 2)
        self.conv3 = Conv2d(64, 128, 3, padding = 2)
        self.conv4 = Conv2d(128, 256, 3, padding = 2)
        self.conv5 = Conv2d(256, 512, 3, padding = 2)
        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn4 = BatchNorm2d(256)
        self.bn5 = BatchNorm2d(512)
        self.pool = MaxPool2d(2, stride = 2)
        self.avg = AvgPool2d(8)
        self.fc = Linear(512*1*1, 2)
        
    def forward(self, x):
        x = self.pool(leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(leaky_relu(self.bn5(self.conv5(x))))
        x = self.avg(x)
        x = x.view(-1, 512*1*1) # flatten
        x = self.fc(x)
        return x

model = SimpleCNN().cuda()

criterion = CrossEntropyLoss()
optimizer = Adamax(model.parameters(), lr = 2e-3)

#############
# Fit model #
#############

num_epochs = 7 

for epoch_i in range(num_epochs):
    model.train() # set train mode
    train_loss = []
    val_loss = []
    
    for batch_i, (images, labels) in enumerate(train_gen):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad() # clear gradients        
        output = model(images)
        loss = criterion(output, labels)
        train_loss.append(loss.item())        
        loss.backward() # compute gradient
        optimizer.step() # update parameters
        
    # ------------- Evaluation on validation data -------------
    model.eval() # set evaluation mode
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(val_gen):
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            loss = criterion(output, labels)
            val_loss.append(loss.item()) 

    print(f"Epoch {epoch_i + 1}/{num_epochs}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}")

##############
# Prediction #
##############

test_df = pd.read_csv("./input/sample_submission.csv")

dataset_test = CancerDataset(test_df, "./input/test/", augmentations = augs_val)
test_gen = DataLoader(dataset_test, batch_size = 32, shuffle = False)

model.eval()
preds = []
with torch.no_grad():
    for batch_i, (images, labels) in enumerate(test_gen):
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        preds_batch = output[:, 1].cpu().numpy() # copy to host memory, convert to numpy
        preds.extend(preds_batch)
        
test_df["label"] = preds

test_df.to_csv("submission.csv", index = False)
