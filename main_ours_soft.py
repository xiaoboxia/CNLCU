from __future__ import print_function
import os
import torch
import tools
import data_closedset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import CNN
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
from loss import loss_ours_soft
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--split_percentage', type = float, help = 'train and validation', default=0.9)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='/output/results_ours_soft/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.3)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, trid, instance]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, or imagenet_tiny', default='svhn')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus,ours]', default='ours')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--co_lambda', type=float, help='sigma^2', default=1e-4)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--channel', type=int, help='channel', default=3)
parser.add_argument('--time_step', type=int, help='time_step', default=3)

args = parser.parse_args()
#
torch.cuda.set_device(args.gpu)

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset=='mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 200
        train_dataset = data_closedset.mnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_closedset.mnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset =  data_closedset.mnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
        
            
    if args.dataset=='fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 200
        train_dataset = data_closedset.fmnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_closedset.fmnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset =  data_closedset.fmnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
        
    if args.dataset=='svhn':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        train_dataset = data_closedset.svhn_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_closedset.svhn_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_closedset.svhn_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target)
        
    if args.dataset=='cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        train_dataset = data_closedset.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_closedset.cifar10_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_closedset.cifar10_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target)
        
    if args.dataset=='cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        train_dataset = data_closedset.cifar100_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_closedset.cifar100_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_closedset.cifar100_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        target_transform=tools.transform_target)
        
    return train_dataset, val_dataset, test_dataset


if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

co_lambda_plan = args.co_lambda * np.linspace(1, 0, args.epoch_decay_start) 
    
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999)


def gen_forget_rate(fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(args.n_epoch) * forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    # if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule


rate_schedule = gen_forget_rate(args.fr_type)

save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, before_loss_1, before_loss_2, sn_1, sn_2, noise_or_not):
    # print('Training %s...' % model_str)
    
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    
    before_loss_1_list=[]
    before_loss_2_list=[]
    
    ind_1_update_list=[]
    ind_2_update_list=[]
    
    
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
        
        start_point = int(i * batch_size)
        stop_point = int((i + 1) * batch_size)

        data = data.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        train_total += 1
        train_correct += prec1

        logits2 = model2(data)
        prec2, = accuracy(logits2, labels, topk=(1,))
        train_total2 += 1
        train_correct2 += prec2
        
        if epoch < args.epoch_decay_start:
        
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean = loss_ours_soft(epoch, before_loss_1[start_point:stop_point], before_loss_2[start_point:stop_point], 
                                                                                                                          sn_1[start_point:stop_point], sn_2[start_point:stop_point], logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, co_lambda_plan[epoch])
        else: 
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean = loss_ours_soft(epoch, before_loss_1[start_point:stop_point], before_loss_2[start_point:stop_point], 
                                                                                                                          sn_1[start_point:stop_point], sn_2[start_point:stop_point], logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, 0.)
        before_loss_1_list += list(np.array(loss_1_mean.detach().cpu()))
        before_loss_2_list += list(np.array(loss_2_mean.detach().cpu()))
        
        ind_1_update_list += list(np.array(ind_1_update + i * batch_size))
        ind_2_update_list += list(np.array(ind_2_update + i * batch_size))
        
        
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, noise_or_not.shape[0]//batch_size, prec1, prec2, loss_1.item(), loss_2.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, before_loss_1_list, before_loss_2_list, ind_1_update_list, ind_2_update_list


# Evaluate the Model
def evaluate(test_loader, model1, model2):
    # print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        
        data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2


def main(args):
    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.seed)
    txtfile = save_dir + "/" + model_str + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
    print(args)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    
    
    noise_or_not = train_dataset.noise_or_not
    
    # Define models
    print('building model...')
    
    
    clf1 = CNN(input_channel=args.channel, n_outputs=args.c)        
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
   
    clf2 = CNN(input_channel=args.channel, n_outputs=args.c)  
    clf2.cuda()
    print(clf2)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 val_acc1 val_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    mean_pure_ratio1=0
    mean_pure_ratio2=0
    
    # evaluate models with random weights
    val_acc1, val_acc2 = evaluate(val_loader, clf1, clf2)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %% Model2 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc1, val_acc2))
    
    test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(val_acc1) + ' ' + str(val_acc2) + ' ' + str(test_acc1) + ' ' + str(test_acc2) +  ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
    val_acc_list = []
    
    test_acc_list = []
    test_acc_list_ = []

    # training
    for epoch in range(0, args.n_epoch):
        if epoch % args.time_step == 0:
            print('Time step initializing...')
            before_loss_1 = 0.0 * np.ones((len(train_dataset), 1))
            before_loss_2 = 0.0 * np.ones((len(train_dataset), 1))
            sn_1 = torch.from_numpy(np.ones((len(train_dataset), 1)))
            sn_2 = torch.from_numpy(np.ones((len(train_dataset), 1)))
        
        
        # train models
        clf1.train()
        clf2.train()
        # scheduler1.step()
        # scheduler2.step()
        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
       
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, before_loss_1_list, before_loss_2_list, ind_1_update_list, ind_2_update_list= train(train_loader, epoch, 
                                                                                                                                                    clf1, optimizer1, clf2, optimizer2, before_loss_1, before_loss_2, sn_1, sn_2, noise_or_not)
        # evaluate models
        val_acc1, val_acc2 = evaluate(val_loader, clf1, clf2)
        val_acc_list.append((val_acc1 + val_acc2) / 2)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        test_acc_list_.append((test_acc1 + test_acc2) / 2)
        # pure_ratio_calculations
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        # save the loss history

        before_loss_1, before_loss_2 = np.array(before_loss_1_list).astype(float), np.array(before_loss_2_list).astype(float)
        # save the selection history
        all_zero_array_1, all_zero_array_2 = np.zeros((len(train_dataset), 1)), np.zeros((len(train_dataset), 1))
        all_zero_array_1[np.array(ind_1_update_list)] = 1
        all_zero_array_2[np.array(ind_2_update_list)] = 1
        print(np.sum(all_zero_array_1))
        sn_1 += torch.from_numpy(all_zero_array_1)
        sn_2 += torch.from_numpy(all_zero_array_2)
        if epoch > 189:
            test_acc_list.append(test_acc1)
            test_acc_list.append(test_acc2)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(val_acc1) + ' ' + str(val_acc2) + ' ' + str(test_acc1) + ' ' + str(test_acc2) +  ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
    id = np.argmax(np.array(val_acc_list))
    test_acc_max = test_acc_list_[id]
    test_acc = np.array(test_acc_list)
    return test_acc_max, np.mean(test_acc)

if __name__ == '__main__':
    best_acc_list = []
    last_acc_list = []
    for i in range(args.n):
        args.seed = i + 1
        args.output_dir = '/output/' + args.d + '/' + str(args.noise_rate) + '/'
        if not os.path.exists(args.output_dir):
            os.system('mkdir -p %s' % (args.output_dir))
        if args.p == 0:
            f = open(args.output_dir + str(args.noise_type) + '_' + str(args.dataset) + '_' + str(
                args.seed) + '.txt', 'a')
            sys.stdout = f
            sys.stderr = f
        best_acc, last_acc = main(args)
        best_acc_list.append(best_acc)
        last_acc_list.append(last_acc)
    print('Best Acc:')
    print(np.array(best_acc_list).mean())
    print(np.array(best_acc_list).std(ddof=1))
    print('Last Ten Acc:')
    print(np.array(last_acc_list).mean())
    print(np.array(last_acc_list).std(ddof=1))
