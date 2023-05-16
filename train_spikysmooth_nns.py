# %%
import copy
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

exec(open("myutils.py").read())
out_dir = 'outputs/nns/'


# define the neural network
class WeightLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_gain=None, init='he'):
        # bias_gain=None -> no biases, else train biases with he or normal initialization with std=bias_gain
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_gain = bias_gain
        self.init=init
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        if init == 'he':
            self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(self.weight.t()) * np.sqrt(2/self.in_features)
        if self.bias_gain is not None:
            x = x + self.bias_gain * self.bias
        return x


class FunctionLayer(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

class FlattenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((x.size()[0],-1))


class AntisymmetricInitializationModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model_1 = model
        self.model_2 = copy.deepcopy(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1/torch.sqrt(2* torch.ones(1)) * (self.model_1(x) - self.model_2(x))

# define the data distribution
def sample_data(n: int):
    x = torch.randn(n, 2)
    x = x / x.norm(dim=-1, keepdim=True)
    y = 1.0 * x[:, 0]
    y = y + 0.5 * torch.randn_like(y)
    return x, y[:, None]

# %%
# define kernel regression
def spikysmooth_gaussians(xdiff, reg, gamma2, gamma1=1):
    return np.exp(-xdiff ** 2 / gamma1) + reg * np.exp(-xdiff ** 2 / gamma2)

def spikysmooth_laplaces(xdiff, reg, gamma2, gamma1=1):
    return np.exp(-xdiff/ gamma1) + reg * np.exp(-xdiff / gamma2)

def spikysmooth_regression(X,y, reg, gamma2, gamma1=1, kernel = 'laplace'):
    n = X.shape[0]
    X_diff = np.zeros((n,n))
    for i in range(n):
        X_diff[i,i] = 0
        for j in range(i):
            X_diff[i,j] = np.linalg.norm(X[i] - X[j])
            X_diff[j,i] = X_diff[i,j]
    if kernel == 'laplace':
        K_matrix = spikysmooth_laplaces(X_diff,reg,gamma2,gamma1)
    elif kernel == 'gaussian':
        K_matrix = spikysmooth_gaussians(X_diff,reg,gamma2,gamma1)
    else:
        raise ValueError('Undefined kernel.')
    kernel_weights = np.linalg.solve(K_matrix,y)
    def regressor(x):
        if len(x.shape) == 1:
            k_vals = np.array([np.linalg.norm(x-X[i]) for i in range(n)])
        elif len(x.shape) == 2:
            m = x.shape[0]
            k_vals = np.zeros((m,n))
            for i in range(m):
                for j in range(n):
                    k_vals[i,j] = np.linalg.norm(x[i]-X[j])
        else:
            raise ValueError('x does not have proper shape.')
        if kernel == 'laplace':
            k_vals = spikysmooth_laplaces(k_vals,reg,gamma2,gamma1)
        elif kernel == 'gaussian':
            k_vals = spikysmooth_gaussians(k_vals,reg,gamma2,gamma1)
        else:
            raise ValueError('Undefined kernel.')
        return k_vals @ kernel_weights
    return regressor, kernel_weights

# sample data and do kernel regression
n_train=15
torch.manual_seed(12)
x, y = sample_data(n_train)
kernel_model,_ = spikysmooth_regression(x,y,reg=0,gamma1=0.4,gamma2=0.01)
kernel_model_spsm,_ = spikysmooth_regression(x,y,reg=1,gamma2=0.01)

angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_linspace = kernel_model(xs_linspace)
ys_spsm_linspace= kernel_model_spsm(xs_linspace)

# %%
# define model training
def train_net_return(n_train: int, width: int, lr: float, n_epochs: int, act: Callable, n_test: int = 10000,
              eval_epochs: int = 1,bias = 0,sgd=False):
    x, y = sample_data(n_train)
    x_test, y_test = sample_data(n_test)
    model = nn.Sequential(WeightLayer(x.shape[-1], width, bias_gain=1), FunctionLayer(act),
                          WeightLayer(width,1, bias_gain=1))
    model = AntisymmetricInitializationModel(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    train_losses = []
    epochs = []
    test_losses = []

    for i_ep in range(n_epochs):
        if (i_ep % eval_epochs == 0 and i_ep <=250) or i_ep % 25 == 0 or i_ep<=15:
            print('.', end='', flush=True)
            epochs.append(i_ep)
            train_losses.append((y - model(x)).square().mean().sqrt().item())
            test_losses.append((y_test - model(x_test)).square().mean().sqrt().item())
            opt.zero_grad(set_to_none=True)
        if sgd:
            idcs = np.random.permutation(x.shape[0])
            for i_sample in idcs:
                y_pred = model(x[[i_sample],:])
                loss = (y[[i_sample]] - y_pred).square().mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        else:
            y_pred = model(x)
            loss = (y - y_pred).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    return model,epochs, train_losses,test_losses


# model hyperparameters
wid = 10000
lr = 4e-2
freq = 100

# run model training
torch.manual_seed(12)
sincos = lambda x: torch.sin(x) + torch.cos(x)
act = lambda x: torch.relu(x) + sincos(freq * x) / (freq)
model_spsm,epochs_spsm, train_spsm, test_spsm =train_net_return(n_train=n_train, width=wid, lr=lr, n_epochs=2500, act=act, eval_epochs=50, n_test=100,bias=2,sgd=True)

torch.manual_seed(12)
model_relu,epochs_relu, train_relu, test_relu =train_net_return(n_train=n_train, width=wid, lr=lr, n_epochs=2500, act=torch.relu, eval_epochs=50, n_test=100,bias=2,sgd=True)

# save train error, test error and function evaluations
savename = f'relu_training_sgd_newscale_n{n_train}_sincos{freq}_width{wid}_ep2500_lr{lr}_heinit_biasgain1_x1_seed{12}_ntest100'

angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_relu = model_relu(xs_linspace)
ys_spsmrelu = model_spsm(xs_linspace)
mysave('outputs/', 'savestats_'+ savename + '.txt',[epochs_spsm, train_spsm, test_spsm,epochs_relu, train_relu, test_relu,ys_relu,ys_spsmrelu])

# %%
# create models with the same weights and the partial activation functions:
sinact = lambda x: sincos(freq * x) / (freq)
sinmodel = nn.Sequential(WeightLayer(x.shape[-1], wid, bias_gain=1), FunctionLayer(sinact), WeightLayer(wid,1, bias_gain=1))
sinmodel = AntisymmetricInitializationModel(sinmodel)

smmodel = nn.Sequential(WeightLayer(x.shape[-1], wid, bias_gain=1), FunctionLayer(torch.relu), WeightLayer(wid,1, bias_gain=1))
smmodel = AntisymmetricInitializationModel(smmodel)

# torch.save(model_spsm,savename+'.pt')
# sinmodel.load_state_dict(torch.load(savename+'.pt'))
# smmodel.load_state_dict(torch.load(savename+'.pt'))
sinmodel.load_state_dict(model_spsm.state_dict())
smmodel.load_state_dict(model_spsm.state_dict())
sinmodel.state_dict()['model_2.2.bias'] = 0
sinmodel.state_dict()['model_1.2.bias'] = 0
sinmodel.eval()
smmodel.eval()

# %%
# create Figure 1

exec(open("plot_helper.py").read())
adjust_fontsize(3)

fig,ax = plt.subplots(1,3, figsize=(3*onefigsize[0], 1 * onefigsize[1]))

# visual plot
angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_linspace = kernel_model(xs_linspace)
ax[0].plot(angles.numpy(), ys_linspace.squeeze(-1),label='Laplace')
ax[0].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
train_angles = torch.atan2(x[:, 1], x[:, 0])
ax[0].plot(train_angles.numpy(), y.squeeze(-1).numpy(), 'x',color='black')
ax[0].plot(angles.numpy(), ys_spsm_linspace.squeeze(-1),label='Spiky-smooth')
ax[0].legend()
ax[0].set_xlabel('Angle (radians)')
ax[0].set_ylabel('Prediction')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_relu = model_relu(xs_linspace)
ys_spsmrelu = model_spsm(xs_linspace)
ys_signal = smmodel(xs_linspace)
ax[1].plot(angles.numpy(), ys_relu.squeeze(-1).detach().numpy(),label='ReLU')
ax[1].plot(angles.numpy(),ys_spsmrelu.squeeze(-1).detach().numpy(),label='Spiky-smooth')
ax[1].plot(angles.numpy(), ys_signal.squeeze(-1).detach().numpy(), label='Signal network',alpha=0.6)
ax[1].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
train_angles = torch.atan2(x[:, 1], x[:, 0])
ax[1].plot(train_angles.numpy(), y.squeeze(-1).numpy(), 'x',color='black')
ax[1].legend()
ax[1].set_xlabel('Angle (radians)')
ax[1].set_ylabel('Prediction')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))


ax[2].plot(epochs_relu, train_relu, label='ReLU',color='tab:blue')
ax[2].plot(epochs_relu, test_relu, '--',color='tab:blue')
ax[2].plot(epochs_spsm, train_spsm, label='Spiky-smooth',color='tab:orange')
ax[2].plot(epochs_spsm, test_spsm, '--',color='tab:orange')
ax[2].set_ylim(bottom=0.0)
ax[2].legend()
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('RMSE')
ax[2].set_xlim(xmin=0,xmax=2480)
xmin, xmax = ax[2].get_xlim()
ax[2].plot([xmin, xmax], [0.5, 0.5], color='black', linestyle = 'dotted')
ax[2].set_xscale('symlog')
enumerate_subplots(ax)
correct_fontsize(ax,has_legend=True,sizes=1.75)
plt.savefig(plot_path+f'jointplot3_laplace{0.4}_'+savename+'_transpsignal_log.pdf') #jointplot3_laplace{0.4}_


# %%
# create figures for Appendix

exec(open("plot_helper.py").read())

adjust_fontsize(2)
fig,ax = plt.subplots(1,2, figsize=(2*onefigsize[0], 1 * onefigsize[1]))

angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_signal = smmodel(xs_linspace)
ys_sp = sinmodel(xs_linspace)
ys_spsmrelu = model_spsm(xs_linspace)
ax[0].plot(angles.numpy(),ys_spsmrelu.squeeze(-1).detach().numpy(),label='Spiky-smooth',color='tab:orange',alpha=0.6)
ax[0].plot(angles.numpy(), ys_signal.squeeze(-1).detach().numpy(),label='Signal network',color='tab:blue')
ax[0].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
train_angles = torch.atan2(x[:, 1], x[:, 0])
ax[0].plot(train_angles.numpy(), y.squeeze(-1).numpy(), 'x',color='black',alpha=0.4)
ax[0].set_xlabel('Angle (radians)')
ax[0].set_ylabel('Prediction')
ax[0].legend()
ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

ax[1].plot(angles.numpy(),ys_sp.squeeze(-1).detach().numpy())
ax[1].plot(angles.numpy(), np.zeros_like(angles.numpy()), 'k--')
ax[1].plot(train_angles.numpy(), y.squeeze(-1).numpy()-np.cos(train_angles.numpy()), 'x',color='black')
ax[1].set_xlabel('Angle (radians)')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

enumerate_subplots(ax)
correct_fontsize(ax,has_legend=False,sizes=1.5)
for item in ax[0].legend().get_texts():
    item.set_fontsize(6*1.5)
plt.savefig(f'spvssm_'+savename+'_withspsm_smallfont.pdf',dpi=300)

# %%
sinact = lambda x: sincos(freq * x) / (freq)
layermodel = nn.Sequential(WeightLayer(x.shape[-1], wid, bias_gain=1), FunctionLayer(sinact))
layermodel = AntisymmetricInitializationModel(layermodel)
thisdict={}
for key in model_spsm.state_dict():
    if '.0.' in key:
        thisdict[key]= model_spsm.state_dict()[key]
layermodel.load_state_dict(thisdict)
layermodel.eval()

# %%
fig,ax = plt.subplots(3,4, figsize=(4*onefigsize[0], 3 * onefigsize[1]))
ys_layer = layermodel(xs_linspace)

def correct_fontsize(axs,has_legend=True, sizes=None):
    # use this for axs already created
    num_ax=len(axs)
    if sizes is None:
        sizes = num_ax
    for i in range(num_ax):
        for item in ([axs[i].title, axs[i].xaxis.label, axs[i].yaxis.label]):
            item.set_fontsize(8*sizes)
        for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            item.set_fontsize(6*sizes)
        if has_legend:
            for item in axs[i].legend().get_texts():
                item.set_fontsize(6*sizes)
    return axs

for i in range(3):
    for j in range(4):
        ax[i,j].plot(angles.numpy(),ys_layer.detach().numpy()[:,i*4+j])
        ax[2,j].set_xlabel('Angle (radians)')
        #ax[1].set_ylabel('Prediction')
        ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax[i,j].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
        ax[i,j].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
correct_fontsize(ax.flatten(),has_legend=False,sizes=3)
plt.savefig('activ_postlayer1_'+savename+'.pdf',dpi=300)


# %%

# recover signal in spsm kernel regression
def signal_spsm_regression(X,y, reg, gamma2, gamma1=1, kernel = 'laplace'):
    n = X.shape[0]
    X_diff = np.zeros((n,n))
    for i in range(n):
        X_diff[i,i] = 0
        for j in range(i):
            X_diff[i,j] = np.linalg.norm(X[i] - X[j])
            X_diff[j,i] = X_diff[i,j]
    if kernel == 'laplace':
        K_matrix = spikysmooth_laplaces(X_diff,reg,gamma2,gamma1)
    elif kernel == 'gaussian':
        K_matrix = spikysmooth_gaussians(X_diff,reg,gamma2,gamma1)
    else:
        raise ValueError('Undefined kernel.')
    kernel_weights = np.linalg.solve(K_matrix,y)
    def regressor(x):
        if len(x.shape) == 1:
            k_vals = np.array([np.linalg.norm(x-X[i]) for i in range(n)])
        elif len(x.shape) == 2:
            m = x.shape[0]
            k_vals = np.zeros((m,n))
            for i in range(m):
                for j in range(n):
                    k_vals[i,j] = np.linalg.norm(x[i]-X[j])
        else:
            raise ValueError('x does not have proper shape.')
        if kernel == 'laplace':
            k_vals_smooth = spikysmooth_laplaces(k_vals,0,gamma2,gamma1)
            k_vals_spiky = reg*spikysmooth_laplaces(k_vals,0,gamma2,gamma2)
        elif kernel == 'gaussian':
            k_vals_smooth = spikysmooth_gaussians(k_vals,0,gamma2,gamma1)
            k_vals_spiky = reg*spikysmooth_gaussians(k_vals,0,gamma2,gamma2)
        else:
            raise ValueError('Undefined kernel.')
        return k_vals_smooth @ kernel_weights, k_vals_spiky @ kernel_weights
    return regressor, kernel_weights


n_train=15
torch.manual_seed(12)
x, y = sample_data(n_train)
signal_kernel_model_spsm,spsm_kernel_weights = signal_spsm_regression(x,y,reg=1,gamma2=0.01)


angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
ys_spsm_signal, ys_spsm_spikes = signal_kernel_model_spsm(xs_linspace)

exec(open("plot_helper.py").read())

adjust_fontsize(2)
fig,ax = plt.subplots(1,2, figsize=(2*onefigsize[0], 1 * onefigsize[1]))

ax[0].plot(angles.numpy(),ys_spsm_signal+ys_spsm_spikes,label='Spiky-smooth',color='tab:orange',alpha=0.6)
ax[0].plot(angles.numpy(), ys_spsm_signal.squeeze(-1),label='Spsm signal',color='tab:blue')
ax[0].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
train_angles = torch.atan2(x[:, 1], x[:, 0])
ax[0].plot(train_angles.numpy(), y.squeeze(-1).numpy(), 'x',color='black',alpha=0.4)
ax[0].set_xlabel('Angle (radians)')
ax[0].set_ylabel('Prediction')
ax[0].legend()
ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

ax[1].plot(angles.numpy(),ys_spsm_spikes.squeeze(-1),label='Spsm spikes')
ax[1].plot(angles.numpy(), np.zeros_like(angles.numpy()), 'k--')
ax[1].plot(train_angles.numpy(), y.squeeze(-1).numpy()-np.cos(train_angles.numpy()), 'x',color='black')
ax[1].set_xlabel('Angle (radians)')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

enumerate_subplots(ax)
correct_fontsize(ax,has_legend=False,sizes=1.5)
for item in ax[0].legend().get_texts():
    item.set_fontsize(6*1.5)
plt.savefig(f'spvssm_laplace_reg{1}_gammas{1}_{0.01}_withspsm_smallfont.pdf',dpi=300)


# %%
# now run the same experiment 100 times:
for irun in range(100):
    torch.manual_seed(12+irun)

    model_spsm,epochs_spsm, train_spsm, test_spsm =train_net_return(n_train=n_train, width=wid, lr=lr, n_epochs=2500, act=act, eval_epochs=50, n_test=10000,bias=2,sgd=True)

    torch.manual_seed(12+irun)
    model_relu,epochs_relu, train_relu, test_relu =train_net_return(n_train=n_train, width=wid, lr=lr, n_epochs=2500, act=torch.relu, eval_epochs=50, n_test=10000,bias=2,sgd=True)

    # save train error, test error and function evaluations
    savename = f'relu_training_sgd_newscale_n{n_train}_sincos{freq}_width{wid}_ep2500_lr{lr}_heinit_biasgain1_x1_seed{12+irun}_ntest10000_dense'

    angles = torch.linspace(-np.pi, np.pi, 3000)
    xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    ys_relu = model_relu(xs_linspace)
    ys_spsmrelu = model_spsm(xs_linspace)

    smmodel = nn.Sequential(WeightLayer(x.shape[-1], wid, bias_gain=1), FunctionLayer(torch.relu), WeightLayer(wid,1, bias_gain=1))
    smmodel = AntisymmetricInitializationModel(smmodel)
    smmodel.load_state_dict(model_spsm.state_dict())
    smmodel.eval()
    ys_signal = smmodel(xs_linspace)
    mysave('outputs/', 'savestats_'+ savename + '_signal.txt',[epochs_spsm, train_spsm, test_spsm,epochs_relu, train_relu, test_relu,ys_relu,ys_spsmrelu,ys_signal])

# %%
statnames=find(f'savestats_relu_training_sgd_newscale_n{n_train}_sincos{freq}_width{wid}_ep2500_lr{lr}_heinit_biasgain1_x1_seed*_ntest10000_dense_signal*','outputs/')

nruns = len(statnames)
epochs_spsm, train_spsm, test_spsm,epochs_relu, train_relu, test_relu,ys_relu,ys_spsmrelu,ys_signal = myload(statnames[0])
all_train_spsm,all_test_spsm, all_train_relu, all_test_relu = [np.zeros((len(train_spsm),nruns)) for _ in range(4)]
all_ys_relu, all_ys_spsmrelu,all_ys_signal = [np.zeros((len(ys_relu),nruns)) for _ in range(3)]

for i,filenam in enumerate(statnames):
    oldepochsspsm=epochs_spsm
    epochs_spsm, train_spsm, test_spsm,epochs_relu, train_relu, test_relu,ys_relu,ys_spsmrelu, ys_signal = myload(filenam)
    if not np.all(epochs_spsm==oldepochsspsm):
        print(oldepochsspsm,epochs_spsm)
    all_train_spsm[:,i] = train_spsm
    all_test_spsm[:,i] =test_spsm
    all_train_relu[:,i] = train_relu
    all_test_relu[:,i] = test_relu
    ys_relu = ys_relu.squeeze(-1).detach().numpy()
    ys_spsmrelu = ys_spsmrelu.squeeze(-1).detach().numpy()
    ys_signal = ys_signal.squeeze(-1).detach().numpy()
    all_ys_relu[:,i] = ys_relu
    all_ys_spsmrelu[:,i] = ys_spsmrelu
    all_ys_signal[:,i] = ys_signal

all_train_spsm = np.sort(all_train_spsm,axis=1)
all_train_relu = np.sort(all_train_relu,axis=1)
all_test_spsm = np.sort(all_test_spsm,axis=1)
all_test_relu = np.sort(all_test_relu,axis=1)
all_ys_spsmrelu = np.sort(all_ys_spsmrelu,axis=1)
all_ys_relu = np.sort(all_ys_relu,axis=1)
all_ys_signal = np.sort(all_ys_signal,axis=1)


# %%
angles = torch.linspace(-np.pi, np.pi, 3000)
xs_linspace = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
all_spsm_signal, all_spsm_spikes, all_ys_kernel = [np.zeros((len(angles),100)) for _ in range(3)]
for iseed in range(100):
    torch.manual_seed(12+iseed)
    x, y = sample_data(n_train)
    signal_kernel_model_spsm,spsm_kernel_weights = signal_spsm_regression(x,y,reg=1,gamma2=0.01)
    kernel_model,_ = spikysmooth_regression(x,y,reg=0,gamma1=0.4,gamma2=0.01)
    ys_spsm_signal, ys_spsm_spikes = signal_kernel_model_spsm(xs_linspace)
    ys_kernel = kernel_model(xs_linspace)
    ys_spsm_signal = ys_spsm_signal.squeeze(-1)
    ys_spsm_spikes = ys_spsm_spikes.squeeze(-1)
    ys_kernel = ys_kernel.squeeze(-1)
    all_spsm_signal[:,iseed] = ys_spsm_signal
    all_spsm_spikes[:,iseed] = ys_spsm_spikes
    all_ys_kernel[:,iseed] = ys_kernel

all_ys_spsm = all_spsm_signal+all_spsm_spikes

all_spsm_signal = np.sort(all_spsm_signal,axis=1)
all_spsm_spikes = np.sort(all_spsm_spikes,axis=1)
all_ys_spsm = np.sort(all_ys_spsm,axis=1)
all_ys_kernel = np.sort(all_ys_kernel,axis=1)    

mysave('outputs/', 'multirunstats_'+ savename + '_signal.txt',[epochs_spsm, all_train_spsm,all_test_spsm, all_train_relu, all_test_relu, all_ys_relu, all_ys_spsmrelu,all_ys_signal,all_spsm_signal,all_spsm_spikes,all_ys_kernel,all_ys_spsm])

# %%
# create Figure 1 with error bars over 100 independent realizations
signif_idx = int(nruns * 0.025)

# if you do just want to load precomputed multirunstats, uncomment:
#nruns=100
#epochs_spsm, all_train_spsm,all_test_spsm, all_train_relu, all_test_relu, all_ys_relu, all_ys_spsmrelu,all_ys_signal,all_spsm_signal,all_spsm_spikes,all_ys_kernel,all_ys_spsm=myload(find('multirunstats_*','outputs/')[0])
#epochs_relu=epochs_spsm

exec(open("plot_helper.py").read())
adjust_fontsize(3)

fig,ax = plt.subplots(1,3, figsize=(3*onefigsize[0], 1 * onefigsize[1]))
ax[2].plot(epochs_relu, all_train_relu.mean(axis=1), label='ReLU',color='tab:blue')
ax[2].fill_between(epochs_relu, all_train_relu[:,signif_idx],all_train_relu[:,-signif_idx-1],alpha=0.25,color='tab:blue')
ax[2].plot(epochs_relu, all_test_relu.mean(axis=1), '--',color='tab:blue')
ax[2].fill_between(epochs_relu, all_test_relu[:,signif_idx],all_test_relu[:,-signif_idx-1],alpha=0.25,color='tab:blue')
ax[2].plot(epochs_spsm, all_train_spsm.mean(axis=1), label='Spiky-smooth',color='tab:orange')
ax[2].fill_between(epochs_spsm, all_train_spsm[:,signif_idx],all_train_spsm[:,-signif_idx-1],alpha=0.25,color='tab:orange')
ax[2].plot(epochs_spsm, all_test_spsm.mean(axis=1), '--',color='tab:orange')
ax[2].fill_between(epochs_spsm, all_test_spsm[:,signif_idx],all_test_spsm[:,-1-signif_idx],alpha=0.25,color='tab:orange')
ax[2].set_ylim(bottom=0.0)
ax[2].set_xlim(xmin=0,xmax=2480)
xmin, xmax = ax[2].get_xlim()
ax[2].plot([xmin, xmax], [0.5, 0.5], color='black', linestyle = 'dotted')
ax[2].legend(loc='upper center')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('RMSE')
ax[2].set_xscale('symlog')

# plot mean functions learned:
ax[1].plot(angles.numpy(), np.mean(all_ys_relu,axis=1),label='ReLU')
ax[1].fill_between(angles.numpy(), all_ys_relu[:,signif_idx],all_ys_relu[:,-signif_idx-1],alpha=0.25,color='tab:blue')
ax[1].plot(angles.numpy(), np.mean(all_ys_spsmrelu,axis=1),label='Spiky-smooth')
ax[1].fill_between(angles.numpy(), all_ys_spsmrelu[:,signif_idx],all_ys_spsmrelu[:,-signif_idx-1],alpha=0.25,color='tab:orange')
ax[1].plot(angles.numpy(), np.mean(all_ys_signal,axis=1),label='Signal')
ax[1].fill_between(angles.numpy(), all_ys_signal[:,signif_idx],all_ys_signal[:,-signif_idx-1],alpha=0.25,color='tab:green')
ax[1].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
ax[1].legend()
ax[1].set_xlabel('Angle (radians)')
ax[1].set_ylabel('Prediction')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

ax[0].plot(angles.numpy(), all_ys_kernel.mean(axis=1),label='Laplace')
ax[0].fill_between(angles.numpy(), all_ys_kernel[:,signif_idx],all_ys_kernel[:,-signif_idx-1],alpha=0.25,color='tab:blue')
ax[0].plot(angles.numpy(), all_ys_spsm.mean(axis=1),label='Spiky-smooth')
ax[0].fill_between(angles.numpy(), all_ys_spsm[:,signif_idx],all_ys_spsm[:,-signif_idx-1],alpha=0.25,color='tab:orange')
ax[0].plot(angles.numpy(), all_ys_signal.mean(axis=1),label='Signal')
ax[0].fill_between(angles.numpy(), all_ys_signal[:,signif_idx],all_ys_signal[:,-signif_idx-1],alpha=0.25,color='tab:green')
ax[0].plot(angles.numpy(), xs_linspace[:, 0].numpy(), 'k--')
ax[0].legend()
ax[0].set_xlabel('Angle (radians)')
ax[0].set_ylabel('Prediction')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
enumerate_subplots(ax)
correct_fontsize(ax,has_legend=True,sizes=1.75)
ax[2].legend(loc='upper center')
for item in ax[2].legend(loc='upper center').get_texts():
    item.set_fontsize(6*1.75)
plt.savefig(f'outputs/figure1_{nruns}runs_alpha0.25.pdf',dpi=300)