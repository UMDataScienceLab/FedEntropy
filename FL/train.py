import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FL.SGD_control_variate import SGD_control_variate
from data import data_to_loader
import matplotlib.pyplot as plt

class CentralizedUpdate(object):
    def __init__(self, args, dataset, criterion="MSELoss"):
        self.args = args
        self.dataset = dataset
        self.trainloader = dataset['train']['loader']
        self.testloader = dataset['test']['loader']
        self.device = 'cuda' if args.gpu else 'cpu'
        # self.loss_criterion = criterion
        
        if self.args.criterion == "MSELoss":
            self.criterion = nn.MSELoss().to(self.device)
        if self.args.criterion == "LogGaussianLoss":
            self.criterion = log_gaussian_loss
        if self.args.loss_penalty:
            self.p_coeff = coeff_exp_decay(
                init=self.args.init_loss_penalty, decay_rate=0.2, dim=self.args.dim_output
            ) #5e-2
            self.criterion = penalty_loss
    
    def update_weights(self, model, epoch, lr, control_variate=None):
        
        # initialize
        model.train()
        batch_loss = []
        
        # set optimizer for updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=self.args.momentum
            )
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-2
            )
        if self.args.optimizer == "sgd_control_variate":
            optimizer = SGD_control_variate(
                model.parameters(), lr=lr, weight_decay=0
            )
        
        # step
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            model.zero_grad()
            outputs = model(inputs)
            
            if not self.args.loss_penalty:            
                if self.args.criterion == "MSELoss":
                    loss = self.criterion(outputs, targets)
                elif self.args.criterion == 'LogGaussianLoss':
                    loss = self.criterion(outputs, targets, torch.exp(model.log_noise))
            else:
                loss = self.criterion(
                    outputs, targets, 
                    loss=self.args.criterion, 
                    sigma=torch.exp(model.log_noise),
                    dim_output=self.args.dim_output, 
                    coeff_penalty=self.p_coeff,
                )

            #; import sys; sys.exit()
                
            loss.backward()
            if control_variate is None:
                optimizer.step()
            else: 
                optimizer.step(
                    c = control_variate['c'],
                    c_i = control_variate['c_i']
                )

            # print
            if self.args.verbose:
                print('Epoch: {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(inputs), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()
                    )
                )
            batch_loss.append(loss.item())
            
        return {
            'model': model,
            'loss': sum(batch_loss)/len(batch_loss)
        }
    
    def get_grad(self, model):
        """
        get gradient for all local data
        """
        grad = {
            n: p.data.zero_() for n, p in copy.deepcopy(model).named_parameters() if p.requires_grad
        }
        model.eval()
        
        for inputs, targets in self.trainloader:
            model.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            
            if not self.args.loss_penalty:            
                if self.args.criterion == "MSELoss":
                    loss = self.criterion(outputs, targets)
                elif self.args.criterion == 'LogGaussianLoss':
                    loss = self.criterion(outputs, targets, torch.exp(model.log_noise))
            else:
                loss = self.criterion(
                    outputs, targets, 
                    loss=self.args.criterion, 
                    sigma=torch.exp(model.log_noise),
                    dim_output=self.args.dim_output, 
                    coeff_penalty=self.p_coeff,
                )
            loss.backward()
            
            for n, p in model.named_parameters():
                grad[n] += p.grad.data / len(self.trainloader)
        
        return grad
        
    
    def inference(self, model, infer='test'):
        
        #initialize
        model.eval()
        
        if infer == "test":
            infer_dataset = self.dataset['test']
        elif infer == 'train':
            infer_dataset = self.dataset['train']
        
        inputs = torch.FloatTensor(infer_dataset['inputs'])
        targets = torch.FloatTensor(infer_dataset['targets'])
        
        outputs = model(inputs)
        
        if not self.args.loss_penalty:
            if self.args.criterion == "MSELoss":
                loss = self.criterion(outputs, targets)
            elif self.args.criterion == 'LogGaussianLoss':
                loss = self.criterion(outputs, targets, torch.exp(model.log_noise))
        else:
            loss = self.criterion(
                    outputs, targets, 
                    loss=self.args.criterion, 
                    sigma=torch.exp(model.log_noise),
                    dim_output=self.args.dim_output, 
                    coeff_penalty=self.p_coeff,
            )     
        
        mseloss = F.mse_loss(outputs, targets[:, :self.args.dim_output])
        
        return {
            "prediction": outputs.detach().numpy(),
            'loss': loss,
            'mseloss': mseloss
        }
    
    def inference_time_window(self, model, eng_idx='all'):
        
        model.eval()
        eng_idx_list = list(self.dataset['test']['targets_bd'])
        
        last_obs = self.args.last_obs
        dim_output = self.args.dim_output
        time_window = self.args.time_window

        std_min, std_max = self.dataset['test']['std_factors']
        
        if eng_idx == 'all':
            
            # prediction 
            pred, mse = {}, {}
            
            for i_eng in eng_idx_list:
                
                last_obs_idx = last_obs - self.args.time_window
                
                x_tw = self.dataset['test']['inputs_bd'][i_eng][last_obs_idx,:][None,:]
                pred_tw = model(torch.FloatTensor(x_tw)).detach().squeeze().numpy()
                obs_tw = self.dataset['test']['data_original'][i_eng][last_obs:(last_obs+dim_output)]
                
                pred[i_eng] = pred_tw
                mse[i_eng] = np.mean(((obs_tw - pred_tw)*(std_max-std_min))**2)
        else: 
            raise NotImplementedError("Eval on only single engine is not implemented yet.")
        
        return {
            "prediction": pred,
            "mse": mse
        }
    
    def plot(self, model, pred_test, title, save_filename=None, plot_train=True):
        
        model.eval()
        
        plt.figure()
        
        YMIN, YMAX = 0, 0.9
        XMIN, XMAX = 0, 50
        n_sample, n_row = 2, 2
        
        idx_train = list(self.dataset['train']['inputs_bd'].keys())[:n_sample]
        idx_test = list(self.dataset['test']['inputs_bd'].keys())[:(n_sample*n_row)]
        
        len_tw = self.args.time_window #list(self.dataset['test']['inputs_bd'].values())[0].shape[1]
        dim_output = self.args.dim_output #list(y_bd_test.values())[0].shape[1]
        last_obs = self.args.last_obs
        
        if plot_train:
        
            for i, i_train in enumerate(idx_train):
            
                x_input = self.dataset['train']['inputs_bd'][i_train]
                sample_y = self.dataset['train']['data_original'][i_train]
                idx_plot = [
                    int(x_input.shape[0]*(1/3)), int(x_input.shape[0]*(2/3)), x_input.shape[0]-1
                ]
                t = range(1,sample_y.shape[0]+1)

                
                for j, i_plot in enumerate(idx_plot):
                    pred_y = model(
                        torch.FloatTensor([x_input[i_plot,:]])
                    ).detach().squeeze().numpy()
                    plt.subplot2grid((6,n_sample), (j, i))
                    plt.plot(t, sample_y, color='blue')
                    plt.plot(
                        t[i_plot:(i_plot+len_tw)], 
                        sample_y[i_plot:(i_plot+len_tw)],
                        color='green'
                    )
                    plt.plot(
                        t[(i_plot+len_tw):(i_plot+len_tw+dim_output)],
                        pred_y, 'r--'
                    )
                    if i == 0: plt.ylabel("Train")
                    if i != 0: plt.yticks([])
                    plt.xticks([])
                    plt.ylim([YMIN, YMAX])
                    plt.xlim([XMIN, XMAX])
                
        m = 0; start = 3 if plot_train else 0 
        for i in range(n_sample):
            for i_row in range(n_row):
                t = range(1, self.dataset['test']['data_original'][idx_test[m]].shape[0]+1)
                pred_y = pred_test['prediction'][idx_test[m]]
                sample_y = self.dataset['test']['data_original'][idx_test[m]]
                plt.subplot2grid((n_row+start,n_sample), (i_row+start, i))
                plt.plot(t[last_obs:], sample_y[last_obs:], color='blue')
                plt.plot(t[:last_obs], sample_y[:last_obs], color='orange')
                plt.plot(t[last_obs:(last_obs+dim_output)], pred_y, 'r--')
                if i ==0 : plt.ylabel('test')
                if i != 0 : plt.yticks([])
                if i_row != n_row-1 : plt.xticks([])
                plt.ylim([YMIN, YMAX])
                plt.xlim([XMIN, XMAX])
                m += 1
        plt.suptitle(title)
    
        if save_filename is not None:
            plt.savefig(save_filename)
        else: 
            plt.show()
        plt.close()
            
    

class LocalUpdate(CentralizedUpdate):
    def __init__(self, args, dataset, client_idx, criterion='MSELoss'):
        super().__init__(args, dataset, criterion)
        self.dataset = self.local_data(
            args=self.args, dataset=dataset, unit_idx=client_idx
        )
        self.trainloader = self.dataset['train']['loader']
        self.testloader = self.dataset['test']['loader']
    
    def local_data(self, args, dataset, unit_idx):
        
        inputs_bd_local = {
            i_unit : dataset['train']['inputs_bd'][i_unit] for i_unit in unit_idx
        }
        targets_bd_local = {
            i_unit : dataset['train']['targets_bd'][i_unit] for i_unit in unit_idx
        }
        inputs_local = np.concatenate([inputs_bd_local[i_unit] for i_unit in unit_idx])
        targets_local = np.concatenate([targets_bd_local[i_unit] for i_unit in unit_idx])
        
        trainloader_local, _, _ = data_to_loader(
            x=inputs_local, y=targets_local, batch_size=self.args.local_bs, 
            shuffle=True
        )
                
        return {
            "train":{
                'loader': trainloader_local,
                'inputs': inputs_local,
                'targets': targets_local,
                'inputs_bd': inputs_bd_local,
                'targets_bd': targets_bd_local,
            },
            "test": dataset['test']
        }
    
    def local_update(self, model, epoch, lr, control_variate=None):
        
        # initialize
        model.train()
        epoch_loss = []
        start_weight = copy.deepcopy(model.state_dict())
        
        # train for local epochs
        for local_epoch in range(self.args.local_ep):
            results_local_update = self.update_weights(model, epoch, lr, control_variate=control_variate)
            epoch_loss.append(results_local_update['loss'])
        
        # calculate gradient
        grad = dict()
        for name, param in results_local_update['model'].named_parameters():
            grad[name] = start_weight[name].detach() - param.detach()
        
        return {
            "model": results_local_update['model'],
            "w": results_local_update['model'].state_dict(),
            "grad": grad,
            "loss": sum(epoch_loss) / len(epoch_loss)
        }


class EWCUpdate(CentralizedUpdate):
    def __init__(self, args, dataset, init_model, grad, criterion='MSELoss'):
        super().__init__(args, dataset, criterion)
        self.trainloader = self.dataset['train']['loader']
        self.testloader = self.dataset['test']['loader']
        self._mean = {
            n: p.data for n, p in copy.deepcopy(init_model).named_parameters() if p.requires_grad
        }
        self._precision = self._diag_fisher(grad)
        self.lambda_penalty = self.args.lambda_penalty
    
    def _diag_fisher(self, grad):
        
        precision_matrices = {
            n: (p_grad**2) for n, p_grad in copy.deepcopy(grad).items()
        }
        
        return precision_matrices
    
    def ewc_penalty(self, model):
        
        loss = 0
        diag_fisher = self._precision
        for n, p in model.named_parameters():
            _loss = diag_fisher[n] * (p - self._mean[n])**2
            loss += _loss.sum()
        return loss
    
    
    def update_weights(self, model, epoch, lr):
        
        # initialize
        model.train()
        batch_loss = []
        
        # set optimizer for updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=self.args.momentum
            )
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-2
            )
        
        # step
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            model.zero_grad()
            outputs = model(inputs)
            
            if not self.args.loss_penalty:            
                if self.args.criterion == "MSELoss":
                    loss = self.criterion(outputs, targets)
                elif self.args.criterion == 'LogGaussianLoss':
                    loss = self.criterion(outputs, targets, torch.exp(model.log_noise))
            else:
                loss = self.criterion(
                    outputs, targets, 
                    loss=self.args.criterion, 
                    sigma=torch.exp(model.log_noise),
                    dim_output=self.args.dim_output, 
                    coeff_penalty=self.p_coeff,
                )
            loss += self.lambda_penalty * self.ewc_penalty(model)
            
            loss.backward()
            optimizer.step()
            
            # print
            if self.args.verbose:
                print('Epoch: {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(inputs), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()
                    )
                )
            batch_loss.append(loss.item())
            
        return {
            'model': model,
            'loss': sum(batch_loss)/len(batch_loss)
        }
    


def log_gaussian_loss(output, target, sigma):
    dim_output = target.shape[1]
    exponent = -0.5*(target - output)**2/(sigma+1e-6)**2
    log_coeff = -dim_output*torch.log(sigma)
    
    return -(log_coeff + exponent).mean()


def penalty_loss(output, target, loss, **kwargs):
    
    dim_output = kwargs['dim_output']
    coeff_penalty = kwargs['coeff_penalty']
    
    l2 = torch.square(output.sub(target[:, dim_output][:,None]))
    penalty = torch.mv(l2, coeff_penalty)
    
    if loss == 'LogGaussianLoss':
        sigma = kwargs['sigma']
        loss = log_gaussian_loss(output, target[:, :dim_output], sigma)
        
    return loss + penalty.mean()
        
    
def coeff_exp_decay(init, decay_rate, dim):
    
    coeff_list = list()
    for i in range(dim):
        val = init*(decay_rate)**(i)
        if val > 1e-6:
            coeff_list.append(val)
        else:
            coeff_list.append(0)
    
    return torch.FloatTensor(coeff_list)



