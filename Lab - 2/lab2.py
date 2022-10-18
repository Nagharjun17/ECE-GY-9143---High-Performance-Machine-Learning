import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import math
import time

class BasicBlock(nn.Module):        #this is the basic blocks of the resnet
    expansion = 1                   #this is used by exercises C2 to C6

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):                                        #this is the main resnet class
    def __init__(self, block, num_blocks, num_classes=10):      #this is used by exercises C2 to C6
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class BasicBlock_nobatch(nn.Module):                            #this is the basic blocks of the resnet without batch normalization layers
    expansion = 1                                               #this is used by exercise C7

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)     #batch normalization removed here for C7
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))                         #batch normalization removed here for C7
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_nobatch(nn.Module):                                            #this is the main resnet class without batch normalization
    def __init__(self, block, num_blocks, num_classes=10):                  #this is used by exercise C7
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)     #batch normalization removed here for C7
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))                       #batch normalization removed here for C7
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])                         #this is used to create the resnet18 model for questions from C2 to C6
#    return ResNet_nobatch(BasicBlock_nobatch, [2, 2, 2, 2])        this is used to create the resnet18 model for the question C7

def train(epoch, net, trainloader, device, optimizer, criterion):    #this function is used by all exercises as this is the training part
    print("\n-------------------------------------------------------------------------------------------------------")
    print('Epoch: ' + str(epoch))
    net.train()         
    train_loss = 0
    correct = 0
    total = 0
    dataloading_time = 0
    training_time = 0
    total_time = 0
    #summary(net,(3,32,32))                              here we print the number of trainable parameters and gradients for Q3
    total_time_start = time.perf_counter()              #timer start for total time for an epoch
    dataload_start = time.perf_counter()                #timer start for dataloading time for an epoch (runs only once here)
    for batch_idx, (inputs, targets) in enumerate(trainloader):         #here we unpack the data and split into batches
        inputs, targets = inputs.to(device), targets.to(device)         #here we load the data to the device
        dataload_end = time.perf_counter()              #timer end for dataloading time for an epoch
        dataloading_time = dataloading_time + dataload_end - dataload_start
        training_start = time.perf_counter()            #timer start for training time for an epoch
        optimizer.zero_grad()                           #clearing the gradients
        outputs = net(inputs)                           #predictions
        loss = criterion(outputs, targets)              #calculate loss
        loss.backward()                                 #backpropagate
        optimizer.step()
        training_end = time.perf_counter()              #timer end for training time for an epoch

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        training_time = training_time + training_end - training_start
        dataload_start = time.perf_counter()            #timer start for dataloading time for the next epoch
    total_time_end = time.perf_counter()                #timer end for total time for an epoch
    total_time = total_time_end - total_time_start
    print("\nLoss: %0.6f, Accuracy: %0.6f" % (train_loss/len(trainloader), 100.*correct/total))         #here we print the loss and accuracy for each epoch from C2 to C7
    print("\nData Loading Time: %0.6f, Training Time: %0.6f, Dataloading + Training Time: %0.6f, Total Time: %0.6f" % (dataloading_time, training_time, dataloading_time+training_time, total_time))    #here we print the calculated dataloading, training and total time for each epoch for C2 to C7
    print("\n-------------------------------------------------------------------------------------------------------")

    return [dataloading_time, training_time, total_time, dataloading_time+training_time]

def main():
  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--cuda', default='n', help='y for gpu and n for cpu')   #this is used to specify the use of GPU or CPU and mainly used in C5
  parser.add_argument('--dataloaders', default=2, type=int, help='number of workers')   #this is used to specify the number of dataloaders and mainly used in C3 and C4
  parser.add_argument('--opt', default='sgd', help='select optimizer')      #this is used to specify the optimizer and mainly used in C6
  parser.add_argument('--datapath', help='specify datapath')
  args = parser.parse_args()
  print("HERE " + str(args.opt) + " OPTIMIZER IS USED")             #this is used to print the optimizer used and mainly used in C6

  transform_train = transforms.Compose([                    #train data preprocessing
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([                     #test data preprocessing
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  trainset = torchvision.datasets.CIFAR10(                                          #download train dataset
      root=args.datapath, train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(                                        #provide train data to dataloader
      trainset, batch_size=128, shuffle=True, num_workers=args.dataloaders)

  testset = torchvision.datasets.CIFAR10(                                           #download test dataset
      root=args.datapath, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(                                         #provide test data to dataloader
      testset, batch_size=100, shuffle=False, num_workers=args.dataloaders)

  classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

  start_epoch = 1
  if(args.cuda):
    if(args.cuda == 'y'):                                           #used cuda if y or use cpu if n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
  else:
    device = 'cpu'

  best_acc = 0

  net = ResNet18()              #loading the model
  net = net.to(device)          #loading it into gpu/cpu
  if device == 'cuda':                      #if gpu, we are parallelizing
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
  criterion = nn.CrossEntropyLoss()
  if(args.opt == 'sgd'):                        #this is used to select the optimizer and mainly used in C6 and Q3
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  elif(args.opt == 'sgdnesterov'):
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
  elif(args.opt == 'adagrad'):
    optimizer = optim.Adagrad(net.parameters(), lr=0.1, weight_decay=5e-4)
  elif(args.opt == 'adadelta'):
    optimizer = optim.Adadelta(net.parameters(), lr=0.1, weight_decay=5e-4)
  elif(args.opt == 'adam'):
    optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

  total_dataloading_time = 0
  total_training_time = 0
  total_training_and_dataloading_time = 0
  total_time_taken = 0
  for epoch in range(start_epoch, start_epoch+5):
    time_definitions = train(epoch, net, trainloader, device, optimizer, criterion)
    total_dataloading_time = total_dataloading_time + time_definitions[0]       #dataloading time
    total_training_time = total_training_time + time_definitions[1]             #training time
    total_time_taken = total_time_taken + time_definitions[2]                   #dataloading time + training time
    total_training_and_dataloading_time = total_training_and_dataloading_time + time_definitions[3]     #total time for an epoch
    scheduler.step()
  print("\n For " + str(args.dataloaders) + " Workers")
  print("\nTotal Data Loading Time: %0.6f, Total Training Time: %0.6f, Total Training + dataloading time = %0.6f, Final Total Time: %0.6f" % (total_dataloading_time, total_training_time, total_training_and_dataloading_time, total_time_taken))  #here we print the calculated total dataloading, total training and final total time for C2 to C7

if __name__=="__main__":
  main()


