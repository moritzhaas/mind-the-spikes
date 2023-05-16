# %%
import numpy as np
import scipy as sp
import os
import mpmath as mp

# run this as a job array on a SLURM cluster or set irun yourself.
irun = int(os.environ['SLURM_ARRAY_TASK_ID'])

mp.dps = 150

# hyperparams
kmax=1000
gamma2s= np.array(mp.matrix(np.array([0.01,0.02,0.025,0.04,0.05,0.075,0.08,0.1,0.15,0.16,0.2,1]))) #newsmallgamma: np.array(mp.matrix(np.array([1e-5,1e-4,0.001,0.005,0.01]))) #smallgamma: np.array(mp.matrix(np.array([0.005,0.01,0.05,0.1,0.5])))
lights= np.linspace(0.5,2,len(gamma2s))
num_rand=3

# set name depending on used gamma2s
smallgamma='paperplot'#'_newsmallgamma' #'_smallgamma' #''
# set name depending on whether the spiky-smooth activations are calculated or only the spike components
mode = 'onlyspikes' #'onlyspikes' # 'spikysmooth'

# set the hyperparameters
if irun == 0:
    gamma0=1
    reg=0.1
elif irun == 1:
    gamma0=1
    reg=1
elif irun == 2:
    gamma0=0.1
    reg = 0.1
elif irun == 3:
    gamma0=0.1
    reg=1
elif irun == 4:
    gamma0=1
    reg=0.1
    xs = np.array(mp.matrix(np.linspace(-3,3,1000)))
elif irun == 5:
    gamma0=1
    reg=1
    xs = np.array(mp.matrix(np.linspace(-3,3,1000)))
elif irun == 6:
    gamma0=0.1
    reg = 0.1
    xs = np.array(mp.matrix(np.linspace(-3,3,1000)))
elif irun == 7:
    gamma0=0.1
    reg=1
    xs = np.array(mp.matrix(np.linspace(-3,3,1000)))
elif irun == 8:
    gamma0=1
    reg=0.1
    xs = np.array(mp.matrix(np.linspace(-4,4,1000)))
elif irun == 9:
    gamma0=1
    reg=1
    xs = np.array(mp.matrix(np.linspace(-4,4,1000)))
elif irun == 10:
    gamma0=0.1
    reg = 0.1
    xs = np.array(mp.matrix(np.linspace(-4,4,1000)))
elif irun == 11:
    gamma0=0.1
    reg=1
    xs = np.array(mp.matrix(np.linspace(-4,4,1000)))



# the following implementations aim to work even for high-order Hermite coefficients by calculating with numbers in reasonable ranges bounded away from 0 and infty
def mp_np(x):
    return np.array(x.tolist(),dtype=float)

def Herm(x,n):
    out = np.array(mp.matrix(len(x),1))
    expr_prev = 1
    expr_curr = x
    for i in range(2,n+1):
        temp = expr_curr
        expr_curr = x * expr_curr - (i-1) * expr_prev
        expr_prev = temp
    return expr_curr

def eval_gauss_rf_activ_iter(x,gamma, pm):
    kmax = len(pm)
    out = np.array(mp.matrix(len(x),1))
    for i in range(kmax):
        prod=1
        for j in range(1,i+1):
            prod *= mp.sqrt(2) / (mp.sqrt(gamma) * j) # one 1/sqrt(i!) from b_i and one 1/sqrt(i!) from h_i
        out = out + pm[i] * prod * Herm(x,i)
    return out * mp.exp(1e2-1/gamma) * mp.exp(-1e2)

def eval_gauss_ntk_activ_iter(x,gamma, pm):
    kmax = len(pm)
    out = np.array(mp.matrix(len(x),1))
    for i in range(kmax):
        prod=1
        for j in range(1,i+1):
            prod *= mp.sqrt(2) / (mp.sqrt(gamma) * j)
        out = out + pm[i] * prod * Herm(x,i) / mp.sqrt(i+1)
    return out * mp.exp(1e2-1/gamma) * mp.exp(-1e2)

def coeff_gauss_rf_activ(gamma,kmax):
    out = np.array(mp.matrix(kmax,1))
    const = mp.exp(1e2-1/gamma) * mp.exp(-1e2)
    prod = 1
    out[0] = const
    for i in range(1,kmax):
        prod *= mp.sqrt(2) / mp.sqrt(gamma * i)
        out[i]=prod * const
    return out

def coeff_gauss_ntk_activ(gamma,kmax):
    out = np.array(mp.matrix(kmax,1))
    const = mp.exp(1e2-1/gamma) * mp.exp(-1e2)
    prod = 1
    out[0] = const
    for i in range(1,kmax):
        prod *= mp.sqrt(2) / mp.sqrt(gamma * i)
        out[i]=prod * const / mp.sqrt(i+1)
    return out

def eval_gauss_rf_spikysmooth_activ_iter(x, gamma, pm, reg, gamma0=1):
    kmax = len(pm)
    out = np.array(mp.matrix(len(x),1))
    prod1 = mp.mpf('1.0')
    prod2 = mp.mpf('1.0')
    for i in range(kmax):
        if i>0:
            prod1 *= 2 / (gamma0 * (i ** 2)) # one 1/sqrt(i!) from b_i and one 1/sqrt(i!) from h_i
            prod2 *= 2 / (gamma * (i ** 2))
        coeff = mp.exp(-1e2) * mp.exp(1e2-1/gamma) * mp.sqrt(mp.exp(2/gamma - 2/gamma0) * prod1 + reg * prod2)
        out = out + pm[i] * coeff * Herm(x,i)
    return out

def eval_gauss_ntk_spikysmooth_activ_iter(x, gamma, pm, reg,gamma0=1):
    kmax = len(pm)
    out = np.array(mp.matrix(len(x),1))
    prod1 = mp.mpf('1.0')
    prod2 = mp.mpf('1.0')
    for i in range(kmax):
        if i>0:
            prod1 *= 2 / (gamma0 * (i ** 2)) # one 1/sqrt(i!) from b_i and one 1/sqrt(i!) from h_i
            prod2 *= 2 / (gamma * (i ** 2))
        coeff = mp.exp(-1e2) * mp.exp(1e2-1/gamma) * mp.sqrt(mp.exp(2/gamma - 2/gamma0) * prod1 / (i+1) + reg * prod2 / (i+1))
        out = out + pm[i] * coeff * Herm(x,i)
    return out

def coeff_gauss_rf_spikysmooth_activ(gamma,kmax,reg,gamma0=1):
    out = np.array(mp.matrix(kmax,1))
    const = mp.exp(-1/(2*gamma)+1e2) * mp.exp(-1e2) * mp.sqrt(mp.exp(1/gamma-2/gamma0-1e2) * mp.exp(1e2)+reg * mp.exp(-1/gamma+1e2)*mp.exp(-1e2))
    out[0]=const
    prod1 = mp.mpf('1.0')
    prod2 = mp.mpf('1.0')
    for i in range(1,kmax):
        prod1 *= 2 / (gamma0 * (i ** 2))
        prod2 *= 2 / (gamma * (i ** 2))
        coeff = mp.sqrt(mp.exp(2/gamma - 2/gamma0-2e2) * mp.exp(2e2) * prod1 + reg * prod2)
        for _ in range((1/gamma) // 1e2): # altogether *exp(-1/gamma)
            coeff = coeff * mp.exp(-1e2)
        coeff = coeff * mp.exp(-(1/gamma)%1e2)
        out[i]=coeff
    return out

def coeff_gauss_ntk_spikysmooth_activ(gamma,kmax,reg,gamma0=1):
    out = np.array(mp.matrix(kmax,1))
    const = mp.exp(-1/(2*gamma)+1e2) * mp.exp(-1e2) * mp.sqrt(mp.exp(1/gamma-2/gamma0-1e2) * mp.exp(1e2)+reg * mp.exp(-1/gamma+1e2)*mp.exp(-1e2))
    out[0]=const
    prod1 = mp.mpf('1.0')
    prod2 = mp.mpf('1.0')
    for i in range(1,kmax):
        prod1 *= 2 / (gamma0 * (i ** 2))
        prod2 *= 2 / (gamma * (i ** 2))
        coeff = mp.sqrt(mp.exp(2/gamma - 2/gamma0-2e2) * mp.exp(2e2) * prod1 / (i+1) + reg * prod2 / (i+1))
        for _ in range((1/gamma) // 1e2): # altogether *exp(-1/gamma)
            coeff = coeff * mp.exp(-1e2)
        coeff = coeff * mp.exp(-(1/gamma)%1e2)
        out[i]=coeff
    return out



import pickle

def mysave(path,name,data):
    if os.path.exists(path):
        if os.path.exists(path+name):
            os.remove(path+name)
        with open(path+name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
    else:
        os.mkdir(path)
        with open(path + name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)

def myload(name):
    with open(name, "rb") as fp:   # Unpickling
        all_stats = pickle.load(fp)
    return all_stats


# %%
# set the desired output folder. Here: 'outputs/activ_evals/'
if irun <= 3:
    rf_coeffslist,ntk_coeffslist = [],[]
    for i,gamma2 in enumerate(gamma2s):
        if mode == 'onlyspikes':
            rf_coeffs = coeff_gauss_rf_activ(gamma2,kmax)
            ntk_coeffs = coeff_gauss_ntk_activ(gamma2,kmax)
        else:
            rf_coeffs = coeff_gauss_rf_spikysmooth_activ(gamma2,kmax,reg,gamma0)
            ntk_coeffs = coeff_gauss_ntk_spikysmooth_activ(gamma2,kmax,reg,gamma0)
        rf_coeffslist.append(rf_coeffs)
        ntk_coeffslist.append(ntk_coeffs)
    if mode == 'onlyspikes':
        mysave('outputs/activ_evals/',f'coeffs_{mode}_rf_ntk_kmax{kmax}_mp{mp.dps}'+smallgamma+'.txt',[gamma2s,rf_coeffslist,ntk_coeffslist])
    else:
        mysave('outputs/activ_evals/',f'coeffs_{mode}_rf_ntk_kmax{kmax}_gamma{gamma0}_reg{reg}_mp{mp.dps}'+smallgamma+'.txt',[gamma2s,rf_coeffslist,ntk_coeffslist])
else:
    np.random.seed(245635)

    for ker in ['ntk','rf']:
        activs1,activsppmm,activsrand = [],[],[]
        pmsrand = []
        for i,gamma2 in enumerate(gamma2s):
            pm1=np.ones(kmax)
            if ker == 'ntk':
                if mode == 'onlyspikes':
                    activ1 = eval_gauss_ntk_activ_iter(xs,gamma2,pm1)
                else:
                    activ1 = eval_gauss_ntk_spikysmooth_activ_iter(xs,gamma2,pm1, reg, gamma0)
            elif ker == 'rf':
                if mode == 'onlyspikes':
                    activ1 = eval_gauss_rf_activ_iter(xs,gamma2,pm1)
                else:
                    activ1 = eval_gauss_rf_spikysmooth_activ_iter(xs,gamma2,pm1, reg, gamma0)
            else:
                raise ValueError(f'Kernel undefined. {ker}')
            
            activs1.append(activ1)

            ppmm = np.ones_like(pm1)
            for j in range(len(ppmm)):
                if np.floor(j / 2) % 2 == 1:
                    ppmm[j] = -1
            if ker == 'ntk':
                if mode == 'onlyspikes':
                    activppmm = eval_gauss_ntk_activ_iter(xs,gamma2,ppmm)
                else:
                    activppmm = eval_gauss_ntk_spikysmooth_activ_iter(xs,gamma2,ppmm, reg, gamma0)
            elif ker == 'rf':
                if mode == 'onlyspikes':
                    activppmm = eval_gauss_rf_activ_iter(xs,gamma2,ppmm)
                else:
                    activppmm = eval_gauss_rf_spikysmooth_activ_iter(xs,gamma2,ppmm, reg, gamma0)
            else:
                raise ValueError(f'Kernel undefined. {ker}')
            
            activsppmm.append(activppmm)

            for _ in range(num_rand):
                pm2 = np.ones_like(pm1)
                rd_idcs = np.where(sp.stats.bernoulli.rvs(0.5, size=len(pm2)) == 1)[0]
                pm2[rd_idcs] = -1
                if ker == 'ntk':
                    if mode == 'onlyspikes':
                        activ2 = eval_gauss_ntk_activ_iter(xs,gamma2,pm2)
                    else:
                        activ2 = eval_gauss_ntk_spikysmooth_activ_iter(xs,gamma2,pm2, reg, gamma0)
                elif ker == 'rf':
                    if mode == 'onlyspikes':
                        activ2 = eval_gauss_rf_activ_iter(xs,gamma2,pm2)
                    else:
                        activ2 = eval_gauss_rf_spikysmooth_activ_iter(xs,gamma2,pm2, reg, gamma0)
                activsrand.append(activ2)
                pmsrand.append(pm2)
        if mode == 'onlyspikes':
            savename = f'activation_{ker}_{mode}_gaussian_mpmath{mp.dps}_iter_kmax{kmax}_x{mp_np(xs).max()}_{len(xs)}_rand{num_rand}_seed{245635}'+smallgamma +'.txt'
        else:
            savename = f'activation_{ker}_{mode}_gaussian_mpmath{mp.dps}_iter_gamma{gamma0}_reg{reg}_kmax{kmax}_x{mp_np(xs).max()}_{len(xs)}_rand{num_rand}_seed{245635}'+smallgamma +'.txt'
        mysave('outputs/activ_evals/',savename,[gamma2s,xs,activs1,activsppmm,activsrand,pmsrand])


# %%
