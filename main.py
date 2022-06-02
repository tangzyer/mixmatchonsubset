from utils import load_data
from MLP import mlp
import torch
import torchvision
from torch.nn import Module
import cifar10 as dataset
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


T = 0.5
K = 4
epochs = 1500
lambda_u = 16
lr = 0.01
label = "lambda"+str(lambda_u)+"K"+str(K)



def sharpen_targets(model, images, augment_K):
    with torch.no_grad():
        p = None
        for i in range(augment_K):
            images_i = images[i]
            outputs_i = model(images_i)
            if p == None:
                p = torch.softmax(outputs_i, dim=1)/augment_K
            else:
                p += torch.softmax(outputs_i, dim=1)/augment_K
        pt = p**(1/T)
        targets = pt / pt.sum(dim=1, keepdim=True)
        targets = targets.detach()
    return targets


def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class LogEntropyLoss(Module):
    # a loss for subset targets data
    def __init__(self):
        super(LogEntropyLoss, self).__init__()

    def forward(self, logits, target):
        # logits format:[[a,b,c,d],[e,f,g,h]]
        # target format:[[1,1,1,0],[0,1,1,0]]
        logits = logits.contiguous()  # [NHW, C]
        logits = F.softmax(logits, 1)
        mask = target.bool()
        logits = logits.masked_fill(~mask, 0)
        loss = torch.sum(logits, dim=1)
        loss = torch.log(loss)
        loss = -1 * loss
        return(loss)

class SemiLoss(object):
    def __call__(self, loss, outputs_x, targets_x_1, targets_x_2, epoch):
        probs_u = torch.softmax(outputs_x, dim=1)
        Lx = torch.mean(loss(outputs_x, targets_x_1))
        Lu = torch.mean((probs_u - targets_x_2)**2)
        return Lx, Lu, lambda_u * linear_rampup(epoch)
        

def train(train_loader, model, optimizer, train_criterion, use_cuda, epoch, ce_loss):
    sum_loss = 0.0
    for data in train_loader:
        (img, data_aug), label = data
        if use_cuda:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            data_aug = [Variable(x).cuda() for x in data_aug]
        else:
            img = Variable(img)
            label = Variable(label)
            data_aug = [Variable(x) for x in data_aug]
        out = model(img)
        outputs_sharpened = sharpen_targets(model, data_aug, K)
        Lx, Lu, w = train_criterion(ce_loss, out, label, outputs_sharpened, epoch)
        entropy_loss = Lx.data.item()
        consistency_loss = (w*Lu).data.item()
        loss = Lx + w * Lu
        print_loss = loss.data.item()
        sum_loss += print_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch', epoch, 'Train Loss:', sum_loss)
    return sum_loss, entropy_loss, consistency_loss

def watch(watch_loader, model, watch_criterion, use_cuda, epoch):
    sum_loss = 0.0
    with torch.no_grad():
        for data in watch_loader:
            (val_inputs, _), label = data
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
            val_outputs = model(val_inputs)
            loss = watch_criterion(val_outputs, label)
            print_loss = loss.data.item()
            sum_loss += print_loss
        print('Epoch:', epoch, 'True Loss:', sum_loss)
    return sum_loss

def valid(valid_loader, model, use_cuda, epoch):
    total_correct = 0.0
    total_num = 0.0
    with torch.no_grad():
        for data in valid_loader:
            (val_inputs, _), label = data
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
            val_outputs = model(val_inputs)
            pred = val_outputs.argmax(dim=1)
            total_correct += torch.eq(pred,label).float().sum().item() #分别为是否相等，scalar tensor转换为float，求和，拿出值
            total_num += label.size(0)
        acc = total_correct/total_num
        print('Epoch:', epoch, 'Val Acc:', acc)
        return acc   

def plot_curve(epochs, train_losses, true_losses, valid_accs, ce_losses, cons_losses, label):
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)
    x4 = range(0, epoch_num)
    x5 = range(0, epoch_num)
    plt.subplot(5, 1, 1)
    plt.plot(x1, valid_accs, 'o-')
    plt.ylabel('Val Acc')
    plt.subplot(5, 1, 2)
    plt.plot(x4, ce_losses, '.-')
    plt.ylabel('Loss 1')
    plt.subplot(5, 1, 3)
    plt.plot(x5, cons_losses, '.-')
    plt.ylabel('Loss 2')
    plt.subplot(5, 1, 4)
    plt.plot(x2, train_losses, '.-')
    plt.ylabel('Train Loss')
    plt.subplot(5, 1, 5)
    plt.plot(x3, true_losses, '.-')
    plt.xlabel('epochs')
    plt.ylabel('True Loss')
    plt.savefig('./logs/'+label +".png")

def log(tag, train_loss, true_loss, val_acc, e_losses, c_losses):
    data = {'train loss':train_loss, "true loss":true_loss, "val acc":val_acc, 'loss term 1':e_losses, 'loss term 2':c_losses}
    df = pd.DataFrame(data)
    df.to_csv('./logs/'+label+tag+'.csv')


def main():
    use_cuda = torch.cuda.is_available()
    print("cuda:",use_cuda)
    model = mlp()
    if use_cuda:
        model = model.cuda()
    train_loader, watch_loader, valid_loader = load_data()
    train_criterion = SemiLoss()
    ce_loss = LogEntropyLoss()
    watch_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    train_losses = []
    true_losses = []
    valid_accs = []
    cons_losses = []
    entropy_losses = []
    for epoch in range(epochs):
        train_loss, entropy_loss, cons_loss = train(train_loader, model, optimizer, train_criterion, use_cuda, epoch, ce_loss)
        entropy_losses.append(entropy_loss)
        cons_losses.append(cons_loss)
        train_losses.append(train_loss)
        true_loss = watch(watch_loader, model, watch_criterion, use_cuda, epoch)
        true_losses.append(true_loss)
        val_acc = valid(valid_loader, model, use_cuda, epoch)
        valid_accs.append(val_acc)
        if epoch % 200 == 1:
            log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
        scheduler.step()
    log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
    plot_curve(epochs, train_losses, true_losses, valid_accs, entropy_losses, cons_losses, label)


if __name__ == '__main__':
    main()
        
        

