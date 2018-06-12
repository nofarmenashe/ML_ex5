import sys
import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ******************* part A *******************
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(16 * 5 * 5, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        # self.nb1 = nn.BatchNorm1d(100)
        # self.nb2 = nn.BatchNorm1d(50)

    def forward(self, x):
        # print("x", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print("before fc: ", x.shape)
        x = x.view(-1, 16 * 5 * 5)
        # print(x.shape)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(dataloaders, optimizer, criterion):
    train_losses = []
    val_losses = []
    accuracy = 0
    epoch = 0

    while accuracy <= 63:
        train_loss = 0.0
        val_loss = 0.0

        for i, data in enumerate(dataloaders['train'], 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for i, data in enumerate(dataloaders['val'], 0):
            inputs, labels = data

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        accuracy = test(dataloaders, dataset_sizes, 'val', criterion)
        val_losses.append(val_loss / dataset_sizes['val'])
        train_losses.append(train_loss / dataset_sizes['train'])
        epoch += 1

    print('Finished Training')
    plot_losses(range(len(train_losses)), train_losses, val_losses)


def test(dataloaders, data_sizes, phase, criterion):
    correct = 0
    total = 0
    sum_loss = 0.0
    dataset_labels = []
    dataset_preds = []
    with torch.no_grad():
        for data in dataloaders[phase]:
            images, labels = data

            # forward
            outputs = net(images)
            current_loss = criterion(outputs, labels)
            sum_loss += current_loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            dataset_labels.extend(labels)
            dataset_preds.extend(predicted)

    accuracy = 100.0 * correct / total
    loss = sum_loss / data_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, accuracy))

    if phase == 'test':
        print("confusion matrix for test: ")
        test_confusion_matrix = metrics.confusion_matrix(dataset_labels, dataset_preds)
        plot_confusion_matrix(test_confusion_matrix)

    return accuracy


def plot_losses(epochs_range, train_losses, validation_losses):
    plt.plot(epochs_range, train_losses, label="train loss")
    plt.plot(epochs_range, validation_losses, label="validation loss")

    plt.show()


def transforms_for_model_A():
    regular_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_transforms = {
        'train': regular_transforms,
        'val': regular_transforms,
        'test': regular_transforms
    }
    return data_transforms


# ******************* part B *******************


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = Variable(inputs.float()), Variable(labels.float())
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloaders, data_sizes, phase, criterion):
    test_loss = 0.0
    test_corrects = 0
    dataset_preds = []
    dataset_labels = []
    for inputs, labels in dataloaders[phase]:

        inputs, labels = Variable(inputs.float()), Variable(labels.float())

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = labels.long()
        loss = criterion(outputs, labels)

        # statistics
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        dataset_preds.extend(preds.data)
        dataset_labels.extend(labels.data)

    loss = test_loss / data_sizes[phase]
    acc = test_corrects.double() / data_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))

    if phase == 'test':
        print("confusion matrix for test: ")
        test_confusion_matrix = metrics.confusion_matrix(dataset_labels, dataset_preds)
        plot_confusion_matrix(test_confusion_matrix)


def transforms_for_model_B():
    regular_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    resize_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_transforms = {
        'train': resize_transforms,
        'val': resize_transforms,
        'test': resize_transforms
    }
    return data_transforms


def load_data(model_transforms, part):
    model_datasets = {x: datasets.CIFAR10(root=os.path.join('./data', part, x), train=x != 'test', download=True,
                                          transform=model_transforms[x]) for x in ['train', 'test']}
    model_datasets['val'] = model_datasets['train']

    num_train = len(model_datasets['train'])
    indices = list(range(num_train))
    split = int(0.2 * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]
    samplers = {
        'train': SubsetRandomSampler(train_idx),
        'val': SubsetRandomSampler(valid_idx),
        'test': None
    }

    model_dataloaders = {x: torch.utils.data.DataLoader(model_datasets[x], batch_size=16, sampler=samplers[x])
                         for x in ['train', 'val', 'test']}

    dataset_sizes = {'train': len(train_idx),
                     'val': len(valid_idx),
                     'test': len(model_datasets['test'])
                     }

    print(dataset_sizes)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return model_dataloaders, classes, dataset_sizes


def plot_confusion_matrix(confusion_matrix):
    # Show confusion matrix in a separate window
    plt.matshow(confusion_matrix, cmap=plt.cm.gray)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':

    part = sys.argv[1]
    if part == 'a':
        model_transoforms = transforms_for_model_A()
    else:
        model_transoforms  =transforms_for_model_B()

    data_loaders, labels_classes, dataset_sizes = load_data(model_transoforms, part)

    if part == 'a':
        net = Net()
        # nn_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        nn_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        nn_criterion = nn.CrossEntropyLoss()
        train(data_loaders, nn_optimizer, nn_criterion)
        test(data_loaders, dataset_sizes, 'train', nn_criterion)
        test(data_loaders, dataset_sizes, 'val' , nn_criterion)
        test(data_loaders, dataset_sizes, 'test', nn_criterion)

    else:
        model_conv = models.resnet18(pretrained=True)
        model_conv.cpu()
        for param in model_conv.parameters():
            param.requires_grad = False

        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 10)
        model_criterion = nn.CrossEntropyLoss()
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.005, momentum=0.7)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        model_conv = train_model(model_conv, model_criterion, optimizer_conv,
                                 exp_lr_scheduler, data_loaders, dataset_sizes, num_epochs=1)
        test_model(model_conv, data_loaders, dataset_sizes, 'train', model_criterion)
        test_model(model_conv, data_loaders, dataset_sizes, 'val', model_criterion)
        test_model(model_conv, data_loaders, dataset_sizes, 'test', model_criterion)
