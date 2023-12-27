import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torchsummary import summary as summary_## 모델 정보를 확인하기 위해 torchsummary 함수 import

## 모델의 형태를 출력하기 위한 함수 
def summary_model(model,input_shape=(3,28,28)):
    model = model.cuda()
    summary_(model, input_shape) ## (model, (input shape))

CFG = {
    'IMG_SIZE':640,
    'EPOCHS':5,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':16,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

all_img_list = glob.glob('./blue/*/*')
train = pd.DataFrame(columns=['img_path', 'label'])
train['img_path'] = all_img_list
train['label'] = train['img_path'].apply(lambda x : str(x).split('/')[-2])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, int(label)
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize((224,224)),
                                    transforms.Normalize(mean, std)])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=6)

def train(model, optimizer, train_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    training_loss = []
    training_acc = []
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        
        running_corrects = 0
        total = 0
        
        for imgs, labels in tqdm(iter(train_loader), desc='Training'):
            imgs = imgs.float().to(device)
            labels = torch.LongTensor(labels).to(device)      # ADDED .type(torch.LongTensor)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            _, preds = torch.max(output, 1)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_corrects += torch.sum(preds == labels.data)
            train_loss.append(loss.item())
            total += preds.size(0)   
                 
        _train_loss = np.mean(train_loss)
        epoch_acc = 100 * (running_corrects.double() / total)
        training_loss.append(_train_loss.item())
        training_acc.append(epoch_acc.item())
        
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Train Acc : [{epoch_acc:.3f} %]')

    x_len = range(1, CFG['EPOCHS']+1)

    plt.plot(x_len, training_acc, label='Training Accuracy')
    plt.plot(x_len, training_loss, label='Training Loss')
    plt.title('Training Accuracy and Training Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('Results.png')
    
    return model

import timm
model_names = timm.list_models(pretrained=True)
print(model_names)

model = timm.create_model('wide_resnet101_2.tv_in1k', pretrained=True, num_classes=2)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])

summary_model(model, (3, 640, 640))

infer_model = train(model, optimizer, train_loader, device)

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in tqdm(iter(train_loader), desc='Testing'):
        output = infer_model(inputs.float().to(device)) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

from sklearn.metrics import confusion_matrix
import seaborn as sn

classes = ["First", "Multiple"]

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
plt.figure()
plt.title('Confusion Matrix on Blue Dataset')
sn.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Confusion_Matrix.png')

print('Done!')