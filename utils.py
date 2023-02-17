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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_lr_finder import LRFinder

def get_metadata_of_CIFAR10_train_dataset():
  exp = datasets.CIFAR10('./data',train =True, download=True)
  exp_data = exp.data
  stat_dict={}
  stat_dict['dataset'] =  exp
  stat_dict['shape'] =  exp_data.shape
  stat_dict['min'] = np.min(exp_data,axis=(0,1,2))/255.
  stat_dict['max'] = np.max(exp_data,axis=(0,1,2))/255.
  stat_dict['mean'] = np.mean(exp_data,axis=(0,1,2))/255.
  stat_dict['std'] = np.std(exp_data,axis=(0,1,2))/255.
  stat_dict['var'] = np.var(exp_data,axis=(0,1,2))/255.
  stat_dict['classes'] = exp.classes
  print('[Train]')
  print('-Numpy Shape :', stat_dict['shape'])
  print('-min:', stat_dict['min'])
  print('-max:', stat_dict['max'])
  print('-mean:',stat_dict['mean'] )
  print('-std:',stat_dict['std'])
  print('-var:', stat_dict['var'])
  print('-classes:', stat_dict['classes'])
  return stat_dict

def visualize_data(dataset,cols=8,rows=5):
  figure=plt.figure(figsize=(14,10))
  for i in range(1,cols*rows+1):
    img,label=dataset[i]
    figure.add_subplot(rows,cols,i)
    plt.title(dataset.classes[label])
    plt.axis("off")
    plt.imshow(img,cmap='gray')
  plt.tight_layout()
  plt.show()

def show_images(dataset,aug_dict,ncol=6):
  nrow=len(aug_dict)
  fig, axes= plt.subplots(ncol,nrow,figsize=(3*nrow,15),squeeze=False)
  for i, (key,aug) in enumerate(aug_dict.items()):
    for j in range(ncol):
      ax = axes[j,i]
      if j ==0:
        ax.text(0.5,0.5,key,horizontalalignment ='center',
                verticalalignment = 'center',fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
      else:
        image,label=dataset[j-1]
        if aug is not None:
          transform = A.Compose([aug])
          image=np.array(image)
          image=transform(image=image)['image']
        
        ax.imshow(image)
        ax.set_title(f'{dataset.classes[label]}')
        ax.axis('off')

  plt.tight_layout()
  plt.show()

class AlbumentationImageDataset(Dataset):
  def __init__(self,image_list,train_agumentation,test_agumentation,train=True):
    self.image_list=image_list
    self.train_agumentation = A.Compose(train_agumentation)
    self.test_agumentation = A.Compose(test_agumentation)
    self.train = train
  def __len__(self):
    return (len(self.image_list))
  def __getitem__(self,i):
    image,label=self.image_list[i]
    if self.train:
      image=self.train_agumentation(image=np.array(image))['image']
    else:
      image=self.test_agumentation(image=np.array(image))['image']
    image = np.transpose(image,(2,0,1)).astype(np.float32)
    return torch.tensor(image,dtype=torch.float),label

def get_CIFAR10_dataset(train_agumentation,test_agumentation,BATCH_SIZE):
  trainset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True)
  testset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True)
  train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset,train_agumentation,test_agumentation,train=True)
  ,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
  test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset,train_agumentation,test_agumentation,train=False)
  ,batch_size=BATCH_SIZE,shuffle=False,num_workers=1)
  return train_loader,test_loader

def get_device():
  use_cuda=torch.cuda.is_available()
  device=torch.device("cuda" if use_cuda else "cpu")
  print('device : ',device)
  return device

def get_summary(device,net):
  summary(net,input_size=(3,32,32))

def find_lr(net,optimizer,criterion,endlr,train_loader):
  lr_finder=LRFinder(net,optimizer,criterion,device="cuda")
  lr_finder.range_test(train_loader, end_lr=endlr, num_iter=118)
  lr_finder.plot()
  min_loss=min(lr_finder.history['loss'])
  ler_rate=lr_finder.history['lr'][np.argmin(lr_finder.history['loss'],axis=0)]
  print("Max LR is {}".format(ler_rate))
  lr_finder.reset() 
  return ler_rate

def get_scheduler(optimizer,data_loader_length,ler_rate):
  return torch.optim.lr_scheduler.OneCycleLR(optimizer
                                                ,max_lr=ler_rate,
                                                steps_per_epoch= data_loader_length,
                                                epochs=24,
                                                pct_start=5/24,
                                                div_factor=10,
                                                three_phase=False,
                                                final_div_factor=50,
                                                anneal_strategy='linear')

def train(model,device,train_loader,optimizer,l1,scheduler):
  model.train()
  pbar=tqdm(train_loader)
  correct=0
  processed=0
  num_loops=0
  train_loss=0
  for batch_idx,(data,target) in enumerate(pbar):
    data,target=data.to(device),target.to(device)
    optimizer.zero_grad()
    y_pred=model(data)
    loss=F.nll_loss(y_pred,target)
    l1=0
    lambda_l1=0.01
    if l1:
      for p in model.parameter():
        l1=l1+p.abs().sum()
    loss = loss+lambda_l1*l1
    loss.backward()
    optimizer.step()
    train_loss+=loss.item()
    scheduler.step()
    pred=y_pred.argmax(dim=1,keepdim=True)
    correct+=pred.eq(target.view_as(pred)).sum().item()
    processed+=len(data)
    num_loops+=1
    pbar.set_description(desc=f'Batch_id={batch_idx} Loss={train_loss/num_loops:.5f} Accuracy={100*correct/processed:0.2f}')
  return 100*correct/processed,train_loss/num_loops

def test(model,device,test_loader):
  model.eval()
  test_loss=0
  correct=0
  with torch.no_grad():
    for data,target in test_loader:
      data,target = data.to(device),target.to(device)
      output=model(data)
      test_loss+=F.nll_loss(output,target,reduction='sum')
      pred=output.argmax(dim=1,keepdim=True)
      correct+=pred.eq(target.view_as(pred)).sum().item()
  test_loss/=len(test_loader.dataset)
  print('\n Test set : Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss,correct,len(test_loader.dataset),
      100.*correct/len(test_loader.dataset)))
  return 100. * correct/len(test_loader.dataset),test_loss

def get_optimizer(net,lr,momentum,l2=False):
  if l2:
    optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum,weight_decay=1e-4)
  else:
    optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum)
  return optimizer

def fit_model(net,device,train_loader,test_loader,scheduler,optimizer,NUM_EPOCHS=20,l1=False,l2=False):
  training_acc,training_loss,testing_acc,testing_loss=list(),list(),list(),list()
  if l2:
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4)
  else:
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
  for epoch in range(1,NUM_EPOCHS+1):
    print("EPOCH:",epoch)
    train_acc,train_loss=train(net,device,train_loader,optimizer,l1,scheduler)
    test_acc,test_loss=test(net,device,test_loader)
    training_acc.append(train_acc)
    training_loss.append(train_loss)
    testing_acc.append(test_acc)
    testing_loss.append(test_loss.item())
  return net,(training_acc,training_loss,testing_acc,testing_loss)
