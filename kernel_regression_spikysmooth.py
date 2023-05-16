# %%
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

mp.dps = 50

kernel = 'laplace' # 'gaussian'

# hyperparams
gamma1=1
gamma2=0.01
reg=1e-1
kmax=100
kmaxs=np.array([5,10,20,30,50])
#gamma2s=np.array([0.0005,0.001,0.005,0.01,0.05,0.1])
gamma2s=np.array(mp.matrix(np.array([0.02,0.04,0.07,0.2,0.5])))
xs = np.array(mp.matrix(np.linspace(-3,3,200)))
lights= np.linspace(0.5,2,len(gamma2s))

exec(open("myutils.py").read())

# %%
# do spiky-smooth kernel regression on finite datasets and plot heatmaps (reg, gamma)
n = 1000
d=2
sigma_noise = np.sqrt(0.5)


# uniform on the sphere
X = np.random.randn(n,d+1)
X /= np.linalg.norm(X,axis=1).reshape((n,1))
#print(np.linalg.norm(xs,axis=1))

# lin, quadratic, sin
ys=np.zeros(n)
noise = sigma_noise * np.random.randn(n)
for i in range(n):
    ys[i] = np.abs(X[i,0]) + X[i,1] ** 2 + np.sin(2*np.pi*X[i,2]) + np.prod(X[i,:]) + noise[i]

# %%
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

# %%
n_test = 10000
X_test = np.random.randn(n_test,d+1)
X_test /= np.linalg.norm(X_test,axis=1).reshape((n_test,1))

# lin, quadratic, sin
ys_test=np.zeros(n_test)
noise_test = sigma_noise * np.random.randn(n_test)
for i in range(n_test):
    ys_test[i] = np.abs(X_test[i,0]) + X_test[i,1] ** 2 + np.sin(2*np.pi*X_test[i,2]) + np.prod(X_test[i,:]) + noise_test[i]

gamma2s = [1e-5,0.001,0.005,0.01,0.05]
regs = [1000,100,10,1, 0.5, 0.1, 0.01]

l2losses=np.zeros((len(gamma2s),len(regs)))
for i,gamma2 in enumerate(gamma2s):
    for j,reg in enumerate(regs):
        regressor, kernel_weights = spikysmooth_regression(X,ys,reg=reg,gamma2=gamma2,gamma1=gamma1,kernel=kernel)
        l2losses[i,j] = ((regressor(X_test) - ys_test) ** 2).mean()

# %%
# Appendix figure
from tueplots import bundles, figsizes

plt.rcParams['text.usetex'] = True
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
plot_path = 'plots/' 

import seaborn as sns
l2excessrisk = l2losses - sigma_noise ** 2
ax = sns.heatmap(l2excessrisk[:,1:], linewidth=0.5,xticklabels=regs[1:],yticklabels=gamma2s,annot=True)
ax.set_title(f'{kernel}, n={n}, d={d}')
ax.set_ylabel(r'Spike bandwidth $\gamma$')
ax.set_xlabel(r'Quasi-regularisation $\rho$')
plt.savefig(plot_path + f'l2excessrisk_heatmap_otherreggamma2_absquadsinpiprod_{kernel}_n{n}_ntest{n_test}_d{d}_sigma{np.round(sigma_noise ** 2,2)}_gammaone{gamma1}.pdf',dpi=300)
plt.show()

# %%
# same for gaussian
l2losses2=np.zeros((len(gamma2s),len(regs)))
for i,gamma2 in enumerate(gamma2s):
    for j,reg in enumerate(regs):
        regressor2, kernel_weights2 = spikysmooth_regression(X,ys,reg=reg,gamma2=gamma2,gamma1=gamma1,kernel='gaussian')
        l2losses2[i,j] = ((regressor2(X_test) - ys_test) ** 2).mean()

l2excessrisk2 = l2losses2 - sigma_noise ** 2

# %%
ax = sns.heatmap(l2excessrisk2[:,1:], linewidth=0.5,xticklabels=regs[1:],yticklabels=gamma2s,annot=True,vmax = 1.1)
ax.set_title(f'gaussian, n={n}, d={d}')
ax.set_ylabel(r'Spike bandwidth $\gamma$')
ax.set_xlabel(r'Quasi-regularisation $\rho$')
plt.savefig(plot_path + f'l2excessrisk_heatmap_otherreggamma2_absquadsinpiprod_gaussian_n{n}_ntest{n_test}_d{d}_sigma{np.round(sigma_noise ** 2,2)}_gammaone{gamma1}.pdf',dpi=300)
plt.show()

# %%
# jointplot
fig,ax = plt.subplots(1,2, figsize=(2*onefigsize[0], 1 * onefigsize[1]))
sns.heatmap(l2excessrisk[:,1:], linewidth=0.5,xticklabels=regs[1:],yticklabels=gamma2s,annot=True, ax=ax[0])
ax[0].set_title(f'{kernel}')
ax[0].set_ylabel(r'Spike bandwidth $\gamma$')
ax[0].set_xlabel(r'Quasi-regularisation $\rho$')

sns.heatmap(l2excessrisk2[:,1:], linewidth=0.5,xticklabels=regs[1:],yticklabels=gamma2s,annot=True,vmax = 1.1,ax=ax[1])
ax[1].set_title(f'gaussian')
ax[1].set_ylabel(r'Spike bandwidth $\gamma$')
ax[1].set_xlabel(r'Quasi-regularisation $\rho$')
plt.savefig(plot_path + f'jointplot_l2excessrisk_heatmap_laplacegaussian_otherreggamma2_absquadsinpiprod_n{n}_ntest{n_test}_d{d}_sigma{np.round(sigma_noise ** 2,2)}_gammaone{gamma1}.pdf',dpi=300)
#plt.show()

mysave(plot_path,f'l2excessrisk_heatmap_laplacegaussian_otherreggamma2_absquadsinpiprod_n{n}_ntest{n_test}_d{d}_sigma{np.round(sigma_noise ** 2,2)}_gammaone{gamma1}.txt',[gamma2s,regs,l2excessrisk,l2excessrisk2])
