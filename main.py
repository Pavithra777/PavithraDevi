import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import utils
import model as m
import copy
import matplotlib.pyplot as plt

train_dataset_dict = utils.get_metadata_of_CIFAR10_train_dataset()
train_dataset = train_dataset_dict['dataset'] 
utils.visualize_data(train_dataset)

train_agumentation = {A.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223 ,0.24348513, 0.26158784)),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1,max_height=8,max_width=8,
                              min_height=8,min_width=8,
                              fill_value= 0.473363,mask_fill_value=None,always_apply=True),
                A.PadIfNeeded(min_height=36, min_width=36, p=0.2,always_apply=True),
                A.RandomCrop(32, 32, always_apply=True, p=0.2)
                }

test_agumentation = {A.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
                }

utils.show_images(train_dataset,{
    'Original Image':None,
    'Horizontal Flip': A.HorizontalFlip(always_apply=True),
    'Cut Out':A.CoarseDropout(max_holes=1,max_height=8,max_width=8,
                              min_height=8,min_width=8,
                              fill_value= 0.473363,mask_fill_value=None,always_apply=True)
    ,'Padded and cropped': A.Compose([A.PadIfNeeded(min_height=36, min_width=36, p=0.2),
                        A.RandomCrop(32, 32, always_apply=True, p=0.2)] ) 
    
})

SEED=1
#CUDA?
cuda=torch.cuda.is_available()
print("CUDA Available:",cuda)

#For reproducibility
torch.manual_seed(SEED)

if cuda:
  torch.cuda.manual_seed(SEED)
  BATCH_SIZE=512
else:
  BATCH_SIZE=512

train_loader,test_loader=utils.get_CIFAR10_dataset(train_agumentation,test_agumentation,BATCH_SIZE)

device= utils.get_device()
print(device)
net=m.CustomResNet().to(device)
utils.get_summary(device,net)
net_exp = copy.deepcopy(net)
from torchvision.transforms import ToTensor
from torch_lr_finder import LRFinder
import torch.nn as nn
from torch_lr_finder import LRFinder


criterion = nn.CrossEntropyLoss()

def find_lr(net,optimizer,criterion,train_loader):
  lr_finder=LRFinder(net,optimizer,criterion,device="cuda")
  lr_finder.range_test(train_loader, end_lr=10, num_iter=118)
  lr_finder.plot()
  min_loss=min(lr_finder.history['loss'])
  ler_rate=lr_finder.history['lr'][np.argmin(lr_finder.history['loss'],axis=0)]
  print("Max LR is {}".format(ler_rate))
  lr_finder.reset() 
  return ler_rate


optimizer= utils.get_optimizer(net_exp,lr=0.001,momentum=0.9,l2=True)
ler_rate = utils.find_lr(net_exp,optimizer,criterion,10,train_loader)
scheduler =utils.get_scheduler(optimizer,len(train_loader),ler_rate)
utils.get_scheduler(optimizer,len(train_loader),ler_rate)

net,history =utils.fit_model(net,device,train_loader,test_loader,scheduler,optimizer,NUM_EPOCHS=24,l1=False,l2=True)



training_acc,training_loss,testing_acc,testing_loss = history

fig, axs = plt.subplots(2,2,figsize=(15,8))
axs[0, 0].plot(training_loss,color='r')
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(training_acc,color='b')
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(testing_loss,color='r')
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(testing_acc,color='b')
axs[1, 1].set_title("Test Accuracy")
leg = axs[0, 0].legend(loc='upper right')
leg = axs[0, 1].legend(loc='upper right')
leg = axs[1, 0].legend(loc='lower right')
leg = axs[1, 1].legend(loc='lower right')
