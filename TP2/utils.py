###################################################################################################################
# Le code suivant a été tiré de: https://github.com/AlexPiche/INF8225/blob/master/tp2/Pytorch%20tutorial.ipynb    #
# Merci à Alexandre Piché pour ces fonctions très utiles.                                                         #
###################################################################################################################

from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from time import time
import numpy as np

def train(model, train_loader, optimizer):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        losses.append(loss.data.item())
        loss.backward()
        optimizer.step()
    return model, losses


def valid(model, valid_loader):
    start_time = time()
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return correct.item() / len(valid_loader.dataset)

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y = []
    y_pred = []
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        y = np.append(y, target.numpy().squeeze())
        y_pred = np.append(y_pred, pred.numpy().squeeze())
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return y, y_pred
    
    
def experiment(model, train_loader, valid_loader, epochs=10, lr=0.001):
    best_precision = 0
    optimizer = Adam(model.parameters(), lr=lr)
    training_time = 0
    validation_time = 0
    training_losses = []
    
    for epoch in range(1, epochs + 1):
        start_time = time()
        model, epoch_losses = train(model, train_loader, optimizer)
        training_losses.extend(epoch_losses)
        training_time += time() - start_time

    start_time = time()
    precision = valid(model, valid_loader)
    validation_time += time() - start_time
    
    print(f"Training time: {training_time//60} min. and {training_time%60} sec.")
    print(f"Validation time: {validation_time//60} min. and {validation_time%60} sec.")

    return model, precision, training_losses
