from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from FL.train import log_gaussian_loss, penalty_loss, coeff_exp_decay

class EWC(object):
    def __init__(self, model: nn.Module, dataset, args):

        self.model = model
        self.dataset = dataset
        self.args = args
        self.device = 'cuda' if args.gpu else 'cpu'
        
        self.trainloader = self.dataset['train']['loader']
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        
        
        # criterion
        if self.args.criterion == "MSELoss":
            self.criterion = nn.MSELoss().to(self.device)
        if self.args.criterion == "LogGaussianLoss":
            self.criterion = log_gaussian_loss
        if self.args.loss_penalty:
            self.p_coeff = coeff_exp_decay(
                init=self.args.init_loss_penalty, decay_rate=0.2, dim=self.args.dim_output
            ) #5e-2
            self.criterion = penalty_loss

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data
    
    def ewc_data(self):
        raise NotImplementedError('not implemented')
    
    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for inputs, targets in self.trainloader:
            self.model.zero_grad()
            inputs = inputs.to(self.device)
            outputs = self.model(input)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.trainloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss_penalty = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss_penalty += _loss.sum()
        return loss_penalty