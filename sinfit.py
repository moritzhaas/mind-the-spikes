# %%
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

mp.dps = 50

# hyperparams
gamma1=0.1
gamma2=0.01
reg=1e-1
kmax=1000
kmaxs=np.array([5,10,20,30,50])
#gamma2s=np.array([0.0005,0.001,0.005,0.01,0.05,0.1])
gamma2s=np.array(mp.matrix(np.array([0.02,0.04,0.07,0.2,0.5])))
n_figs=len(gamma2s)
xs = np.array(mp.matrix(np.linspace(-3,3,200)))
lights= np.linspace(0.5,2,len(gamma2s))
lights2= np.linspace(0.5,1.75,3)
num_rand=3
n_figs=len(gamma2s)


exec(open("myutils.py").read())

exec(open("plot_helper.py").read())

adjust_fontsize(3)



# %%
def myntkfit(x,gamma):
    return np.sqrt(gamma) * np.sin(np.sqrt(2)*x/np.sqrt(gamma)+np.pi/4)

def myrffit(x,gamma):
    return np.sqrt(2) * np.sin(np.sqrt(2)*x/np.sqrt(gamma)+np.pi/4)


# the NNGP activation overlaps very well with its approximation
fig,ax = plt.subplots()
for filenam in find(f'activation_rf_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    xs = np.array(xs.tolist(), dtype=float)
    for ifig, i in enumerate([4,2,0]):
        gamma2=np.array(gamma2s[i],dtype=np.float64)
        theseactivs=activsppmm[i][np.where(xs>-0.5)[0]]
        idx0=np.where(theseactivs>0)[0][0]
        theseactivs[np.where(theseactivs>0)[0][0]]
        x0=xs[np.where(xs>-0.5)[0]][idx0]
        ax.plot(xs,activsppmm[i],label = f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
        ax.plot(xs, myrffit(xs,gamma2),color= adjust_lightness('tab:orange', amount=lights2[::-1][ifig]))
ax.plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax.set_xlim(-1.5,1.5)


# %%

# plot nngpfit:
fig,ax = plt.subplots(1,2,figsize=(2*onefigsize[0], 1 * onefigsize[1]))

for filenam in find(f'activation_rf_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    xs = np.array(xs.tolist(), dtype=float)
    for ifig, i in enumerate([6,4,2]):
        gamma2=np.array(gamma2s[i],dtype=np.float64)
        #np.where(activsppmm[4]>0)
        theseactivs=activsppmm[i][np.where(xs>-0.5)[0]]
        idx0=np.where(theseactivs>0)[0][0]
        theseactivs[np.where(theseactivs>0)[0][0]]
        x0=xs[np.where(xs>-0.5)[0]][idx0]
        ax[0].plot(xs,np.abs(activsppmm[i]-myrffit(xs,gamma2)),label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[0].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[0].set_xlim(-2.5,2.5)
ax[0].set_yscale('log')
ax[0].set_ylabel('Absolute error')
ax[0].legend(loc='lower left')
ax[0].set_xlabel('NNGP')

# plot ntkfit:

for filenam in find(f'activation_ntk_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    xs = np.array(xs.tolist(), dtype=float)
    for ifig, i in enumerate([5,3,0]):
        gamma2=np.array(gamma2s[i],dtype=np.float64)
        theseactivs=activsppmm[i][np.where(xs>-0.5)[0]]
        idx0=np.where(theseactivs>0)[0][0]
        theseactivs[np.where(theseactivs>0)[0][0]]
        x0=xs[np.where(xs>-0.5)[0]][idx0]
        ax[1].plot(xs,np.abs(activsppmm[i]-myntkfit(xs,gamma2)),label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[1].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[1].set_xlim(-2.5,2.5)
ax[1].set_yscale('log')
ax[1].set_xlabel('NTK')
ax[1].legend(loc='lower right')
enumerate_subplots(ax, pos_x=-0.08, pos_y=1.05, fontsize=16)
plt.savefig(plot_path+f'activationplots/approxerrorenum_kmax{1000}_mp{150}_gammalegend.pdf')


# %%

# plot difference between spikysmooth and smooth activation fct, and plot onlyspikes, and plot fitted sin, all the same?
filenam_spsm=find('activation_ntk_spikysm*_gaussian_mpmath150_iter_gamma1_reg1_kmax1000_x4.0*paper*','outputs/')[0]
filenam_smooth=find('activation_ntk_onlysmooth*_gaussian_mpmath150_iter*_kmax1000_x4.0*paper*','outputs/')[0]
filenam_sp=find('activation_ntk_onlysp*_gaussian_mpmath150_iter*_kmax1000_x4.0*paper*','outputs/')[0]

gamma2s_sp,xs_sp,activs1_sp,activsppmm_sp,activsrand_sp,pmsrand_sp = myload(filenam_sp)
gamma2s_spsm,xs_spsm,activs1_spsm,activsppmm_spsm,activsrand_spsm,pmsrand_spsm = myload(filenam_spsm)
gamma2s_sm,xs_sm,activs1_sm,activsppmm_sm,activsrand_sm,pmsrand_sm = myload(filenam_smooth)

# smooth means onlyspike with gamma = gamma0 = 1
gamma2s_sm,activs1_sm,activsppmm_sm,activsrand_sm,pmsrand_sm =gamma2s_sm[0],activs1_sm[0],activsppmm_sm[0],activsrand_sm[0],pmsrand_sm[0]

if not np.all(gamma2s_spsm == gamma2s_sp):
    raise ValueError('Gammas dont match.')
if not (np.all(xs_sm == xs_sp) and np.all(xs_sm == xs_spsm)):
    raise ValueError('xs dont match.')
xs = xs_sm
xs=np.array(xs.tolist(), dtype=np.float64)

# now plot spsm-sm, spikes and fitted sin
n_figs=4
idcs=[4,3,1,0]
fig,ax = plt.subplots(1,n_figs, figsize=(n_figs*onefigsize[0],1*onefigsize[1]))
for iplt, ig in enumerate(idcs):
    gamma2_spsm = gamma2s_spsm[ig]
    ax[iplt].plot(xs, activsppmm_spsm[ig]-activsppmm_sm, label='spsm-sm')
    ax[iplt].plot(xs, activsppmm_sp[ig],label='spike')
    ax[iplt].plot(xs, myntkfit(xs,np.array(gamma2_spsm,dtype=float)),label='sin')
    ax[iplt].set_xlabel(r'$\gamma$='+f'{gamma2_spsm}')
plt.legend()
correct_fontsize(ax,has_legend=False,sizes=2)
for item in ax[-1].legend().get_texts():
    item.set_fontsize(6*2)
plt.savefig('outputs/additive_decomp_analyticcosfit_' + filenam_spsm[:-4].split('/',10)[-1] + '.pdf',dpi=300)

# %%
# how bad is the sin error:

fig,ax = plt.subplots(1,n_figs, figsize=(n_figs*onefigsize[0],1*onefigsize[1]))
for iplt, ig in enumerate(idcs):
    gamma2_spsm = gamma2s_spsm[ig]
    ax[iplt].plot(xs, activsppmm_spsm[ig]-activsppmm_sm-activsppmm_sp[ig], label='spsm-(sm+sp)')
    ax[iplt].plot(xs, activsppmm_spsm[ig]-activsppmm_sm-myntkfit(xs,np.array(gamma2_spsm,dtype=float)), label='spsm-(sm+sin)')
    ax[iplt].set_xlabel(r'$\gamma$='+f'{gamma2_spsm}')
plt.legend()
correct_fontsize(ax,has_legend=False,sizes=2)
for item in ax[-1].legend().get_texts():
    item.set_fontsize(6*2)
plt.savefig('outputs/error_additive_decomp_analyticcosfit_' + filenam_spsm[:-4].split('/',10)[-1] + '.pdf',dpi=300)
# %%
