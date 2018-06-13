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
import random


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(16 * 5 * 5, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, dataloaders, optimizer, criterion):
    train_losses = []
    validation_losses = []
    done_training = False

    while not done_training:
        train_loss = 0.0
        validation_loss = 0.0

        for i, data in enumerate(dataloaders['train'], 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for i, data in enumerate(dataloaders['validation'], 0):
            inputs, labels = data

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

        train_losses.append(train_loss / dataset_sizes['train'])
        validation_losses.append(validation_loss / dataset_sizes['validation'])

        total, correct, sum_loss, labels, predictions = test_set('validation', dataloaders, criterion)
        validation_accuracy = 100.0 * correct / total
        done_training = validation_accuracy > 63.5

    return train_losses, validation_losses


def test_set(set, dataloaders, criterion):
    correct = 0
    total = 0
    sum_loss = 0.0

    all_labels = []
    all_predictions = []

    for data in dataloaders[set]:
        images, labels = data

        # forward
        outputs = net(images)
        current_loss = criterion(outputs, labels)
        sum_loss += current_loss.item()
        _, predicted = torch.max(outputs.data, 1)

        # statistics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels)
        all_predictions.extend(predicted)

    return total, correct, sum_loss, all_labels, all_predictions


def test(dataloaders, data_sizes, criterion):
    labels_to_plot, predictions_to_plot = [], []

    with torch.no_grad():
        for set in ['train', 'validation', 'test']:
            total, correct, sum_loss, labels, predictions = test_set(set, dataloaders, criterion)

            if set == 'test':
                labels_to_plot = labels
                predictions_to_plot = predictions

            accuracy = 100.0 * correct / total
            loss = sum_loss / data_sizes[set]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(set, loss, accuracy))

    return labels_to_plot, predictions_to_plot


def plot_losses(num_of_epochs, train_losses, validation_losses):
    epochs = range(num_of_epochs)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.show()


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation set
        for set in ['train', 'validation']:
            if set == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[set]:
                inputs, labels = Variable(inputs.float()), Variable(labels.float())
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(set == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)

                    if set == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[set]
            epoch_acc = running_corrects.double() / dataset_sizes[set]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(set, epoch_loss, epoch_acc))

            # deep copy the model
            if set == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def model_test_set(model, set, dataloaders, criterion):
    correct = 0
    total = 0
    sum_loss = 0.0

    all_labels = []
    all_predictions = []

    for inputs, labels in dataloaders[set]:
        inputs, labels = Variable(inputs.float()), Variable(labels.float())

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = labels.long()
        loss = criterion(outputs, labels)

        # statistics
        total += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

        all_labels.extend(labels.data)
        all_predictions.extend(preds.data)

    return total, correct, sum_loss, all_labels, all_predictions


def test_model(model, dataloaders, data_sizes, criterion):

    labels_to_plot, predictions_to_plot = [], []

    for set in ['train', 'validation', 'test']:
        total, correct, sum_loss, labels, predictions = model_test_set(model, set, dataloaders, criterion)

        loss = total / data_sizes[set]
        accuracy = correct.double() / data_sizes[set]

        if set == 'test':
            labels_to_plot = labels
            predictions_to_plot = predictions

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(set, loss, accuracy))

    return labels_to_plot, predictions_to_plot


def get_transformations(question):
    if question == '1':
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return transformations


def load_data(question):
    tranformations = get_transformations(question)
    model_datasets = {x: datasets.CIFAR10(root=os.path.join('./data', question, x), train=x != 'test', download=True,
                                          transform=tranformations) for x in ['train', 'test']}
    model_datasets['validation'] = model_datasets['train']

    train_set_size = len(model_datasets['train'])
    indices = list(range(train_set_size))
    random.shuffle(indices)
    split = int(0.2 * train_set_size)
    train_idx, validation_idx = indices[split:], indices[:split]

    samplers = {
        'train': SubsetRandomSampler(train_idx),
        'validation': SubsetRandomSampler(validation_idx),
        'test': None
    }

    model_dataloaders = {x: torch.utils.data.DataLoader(model_datasets[x], batch_size=32, sampler=samplers[x])
                         for x in ['train', 'validation', 'test']}

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_sizes = {'train': len(train_idx),
                     'validation': len(validation_idx),
                     'test': len(model_datasets['test'])
                     }
    
    return model_dataloaders, classes, dataset_sizes


def generate_confusion_matrix(labels, predictions):
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    plt.matshow(confusion_matrix, cmap=plt.cm.BuPu_r)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def write_results_to_file(file_name, predictions):
    with open(file_name, "w") as file:
        for i, x in enumerate(predictions):
            file.write(str(x.item()))
            if i != len(predictions) - 1:
                file.write("\n")


if __name__ == '__main__':

    question = sys.argv[1]

    data_loaders, labels_classes, dataset_sizes = load_data(question)

    if question == '1':
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_losses, validation_losses = train(net, data_loaders, optimizer, criterion)
        num_of_epochs = len(train_losses)
        plot_losses(num_of_epochs, train_losses, validation_losses)

        labels, predictions = test(data_loaders, dataset_sizes, criterion)
        generate_confusion_matrix(labels, predictions)

        write_results_to_file("test_pred", predictions)

    else:
        model = models.resnet18(pretrained=True)
        model.cpu()
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr=0.003, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        model = train_model(model, criterion, optimizer, exp_lr_scheduler, data_loaders, dataset_sizes, num_epochs=1)

        labels, predictions = test_model(model, data_loaders, dataset_sizes, criterion)
        generate_confusion_matrix(labels, predictions)
