from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import LocalOutlierFactor
#from pyod.models.knn import KNN
import warnings

warnings.filterwarnings('ignore')


def soft_process(loss):
    # loss: float(tensor)
    # loss: array / tensor array 
    soft_loss = torch.log(1+loss+loss*loss/2)
    return soft_loss

def loss_ours_soft(epoch, before_loss_1, before_loss_2, sn_1, sn_2, y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda):
    # before_loss: the mean of soft_losses with size: batch_size * 1
    # co_lambda: sigma^2
    # sn_1, sn_2: selection number 
    before_loss_1, before_loss_2 = torch.from_numpy(before_loss_1).cuda().float(), torch.from_numpy(before_loss_2).cuda().float()
    
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()
    
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    
    loss_1 = soft_process(loss_1)
    
    loss_1_mean = (before_loss_1 * s + loss_1) / (s + 1)
    confidence_bound_1 = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn_1 + 1) - co_lambda)
    soft_criterion_1 = F.relu(loss_1_mean.float() - confidence_bound_1.cuda().float())
        
    ind_1_sorted = np.argsort(soft_criterion_1.cpu().data).cuda()
    soft_criterion_1_sorted = soft_criterion_1[ind_1_sorted]
    
   
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    loss_2 = soft_process(loss_2)
    
    loss_2_mean = (before_loss_2 * s + loss_2) / (s + 1)
    confidence_bound_2 = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn_2 + 1) - co_lambda)
    soft_criterion_2 = F.relu(loss_2_mean.float() - confidence_bound_2.cuda().float())
    ind_2_sorted = np.argsort(soft_criterion_2.cpu().data).cuda() 
    
    soft_criterion_2_sorted = soft_criterion_2[ind_2_sorted]
                                      
                                      
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(soft_criterion_1_sorted))
    
    # index for updates
    ind_1_update = ind_1_sorted[0][:num_remember].cpu()
    ind_2_update = ind_2_sorted[0][:num_remember].cpu()
    
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[0][:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[0][:num_remember]]])/float(num_remember)
    
 
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean



def hard_process(loss):
    # loss: numpy_matrix
    # return: loss_matrix
    loss = loss.detach().cpu().numpy()
    dim_1, dim_2 = loss.shape[0], loss.shape[1]
    if dim_2 >= 5:
        lof = LocalOutlierFactor(n_neighbors=2, algorithm='auto', contamination=0.1, n_jobs=-1)
        #lof = KNN(n_neighbors=2)
        t_o = []
        for i in range(dim_1):
            loss_single = loss[i].reshape((-1, 1))
            outlier_predict_bool = lof.fit_predict(loss_single)
            outlier_number = np.sum(outlier_predict_bool>0)
            loss_single[outlier_predict_bool==1] = 0.
            loss[i,:] = loss_single.transpose()
            t_o.append(outlier_number)
        t_o = np.array(t_o).reshape((dim_1, 1))
    else:
        t_o = np.zeros((dim_1, 1))
    loss = torch.from_numpy(loss).cuda().float()
    return loss, t_o



def loss_ours_hard(epoch, before_loss_1, before_loss_2, sn_1, sn_2, y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda, loss_bound):
    # before_loss: the losses with size: batch_size * time_step type numpy
    # co_lambda: t_min
    # sn_1, sn_2: selection number 

  
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()
    loss_bound = torch.tensor(loss_bound).float()
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
 
    before_and_loss_1 = torch.cat((torch.from_numpy(before_loss_1).cuda().float(), loss_1.unsqueeze(1).float()), 1)

    before_and_loss_1_hard, t_o_1 = hard_process(before_and_loss_1)
    loss_1_mean = torch.mean(before_and_loss_1_hard, dim=1)
    confidence_bound_1_list = []
    for i in range(loss_1_mean.shape[0]):
        confidence_bound_1 = 2 * torch.sqrt(2 * co_lambda) * loss_bound * (s + 1.414 * torch.tensor(t_o_1[i]).double()) * torch.sqrt(torch.log(4*s)/sn_1[i]) / ((s - torch.tensor(t_o_1[i]).double()) * torch.sqrt(s))
        confidence_bound_1_list.append(confidence_bound_1)
        
    confidence_bound_1_numpy = torch.from_numpy(np.array(confidence_bound_1_list)).float().cuda()
    
    hard_criterion_1 = F.relu(loss_1_mean - confidence_bound_1_numpy)
      
    ind_1_sorted = np.argsort(hard_criterion_1.cpu().data).cuda()
    hard_criterion_1_sorted = hard_criterion_1[ind_1_sorted]
    
   
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
 
    before_and_loss_2 = torch.cat((torch.from_numpy(before_loss_2).cuda().float(), loss_2.unsqueeze(1).float()), 1)

    before_and_loss_2_hard, t_o_2 = hard_process(before_and_loss_2)
    loss_2_mean = torch.mean(before_and_loss_2_hard, dim=1)
    confidence_bound_2_list = []
    for i in range(loss_2_mean.shape[0]):
        confidence_bound_2 = 2 * torch.sqrt(2 * co_lambda) * loss_bound * (s + 1.414 * torch.tensor(t_o_2[i]).double()) * torch.sqrt(torch.log(4*s)/sn_2[i]) / ((s - torch.tensor(t_o_2[i]).double()) * torch.sqrt(s))
        confidence_bound_2_list.append(confidence_bound_2)
        
    confidence_bound_2_numpy = torch.from_numpy(np.array(confidence_bound_2_list)).float().cuda()
    
    hard_criterion_2 = F.relu(loss_2_mean - confidence_bound_2_numpy)
      
    ind_2_sorted = np.argsort(hard_criterion_2.cpu().data).cuda()
    hard_criterion_2_sorted = hard_criterion_2[ind_2_sorted]
                                      
                                      
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(hard_criterion_1_sorted))
    
    # index for updates
    ind_1_update = ind_1_sorted[:num_remember].cpu().numpy()
    ind_2_update = ind_2_sorted[:num_remember].cpu().numpy()
    
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)
    
 
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, loss_1, loss_2



