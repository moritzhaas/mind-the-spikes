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
# Figure 3: a. NTK Hermite series, b. NTK ++-- activation, c. NTK random sign activations

filename_coeff_largegamma = find('coeffs_*kmax400*mp100.txt', 'outputs/')[0]

gamma2s,rf_coeffslist,ntk_coeffslist = myload(filename_coeff_largegamma)

fig,ax = plt.subplots(1,3, figsize=(3*onefigsize[0], 1 * onefigsize[1]))

for j,gamma2 in enumerate(gamma2s[::-1][1:]):
    absol_val_rf = rf_coeffslist[::-1][j+1]
    absol_val_ntk = ntk_coeffslist[::-1][j+1]
    ax[2].scatter(np.arange(len(absol_val_ntk)),absol_val_ntk,label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights[::-1][j+1]))

ax[2].set_xlabel('Index')
ax[2].set_ylabel('Coefficient value')
ax[2].legend()
ax[2].set_xlim(-5,180)

for filenam in find(f'activation_ntk_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    for ifig, i in enumerate([10,7,4]):
        gamma2=gamma2s[i]
        ax[0].plot(xs,activsppmm[i],label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[0].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[0].plot([0,0],ax[0].get_ylim(),color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[0].set_xlim(-2.5,2.5)
ax[0].legend(loc='upper left')

for filenam in find(f'activation_ntk_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    for ifig, i in enumerate([10,7,4]):
        gamma2=gamma2s[i]
        if i == 10:
            idx=i*3+1
        else:
            idx=i*3
        ax[1].plot(xs,activsrand[idx],label = f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[1].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[1].plot([0,0],ax[1].get_ylim(),color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[1].set_xlim(-3.01,3.01)
ax[1].set_ylim(-1.02,1.02)

enumerate_subplots(ax)
correct_fontsize(ax,has_legend=False,sizes=2)
for item in ax[0].legend().get_texts():
    item.set_fontsize(6*2)
for item in ax[2].legend().get_texts():
    item.set_fontsize(6*2)
plt.savefig(plot_path+f'activationplots/jointplot_fig3_ntk_kmax{kmax}_mp{150}_gammalegend.pdf')


# %%
# Figure 3 for the NNGP
gamma2s,rf_coeffslist,ntk_coeffslist = myload(filename_coeff_largegamma)

fig,ax = plt.subplots(1,3, figsize=(3*onefigsize[0], 1 * onefigsize[1]))

for j,gamma2 in enumerate(gamma2s[::-1][1:]):
    absol_val_rf = rf_coeffslist[::-1][j+1]
    absol_val_ntk = ntk_coeffslist[::-1][j+1]
    ax[2].scatter(np.arange(len(absol_val_rf)),absol_val_rf,label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights[::-1][j+1]))

ax[2].set_xlabel('Index')
ax[2].set_ylabel('Coefficient value')
ax[2].legend()
ax[2].set_xlim(-5,180)

for filenam in find(f'activation_rf_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    for ifig, i in enumerate([10,7,4]):
        gamma2=gamma2s[i]
        ax[0].plot(xs,activsppmm[i],label = r'$\gamma$ = '+f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[0].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[0].plot([0,0],ax[0].get_ylim(),color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[0].set_xlim(-2.5,2.5)
ax[0].legend(loc='lower left')

for filenam in find(f'activation_rf_onlyspikes*mpmath150_*kmax{kmax}_x3*paperplot.txt','outputs/'):
    gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
    for ifig, i in enumerate([10,7,4]):
        gamma2=gamma2s[i]
        if i == 10:
            idx=i*3+1
        else:
            idx=i*3
        ax[1].plot(xs,activsrand[idx],label = f'{gamma2}', color=adjust_lightness('tab:blue', amount=lights2[::-1][ifig]))
ax[1].plot([-3,3],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[1].plot([0,0],ax[1].get_ylim(),color = 'black',linewidth =1,alpha=0.4,zorder=-10)
ax[1].set_xlim(-3.01,3.01)
ax[1].set_ylim(-4.01,4.01)

enumerate_subplots(ax)
correct_fontsize(ax,has_legend=False,sizes=2)
for item in ax[0].legend().get_texts():
    item.set_fontsize(6*2)
for item in ax[2].legend().get_texts():
    item.set_fontsize(6*2)
#ax[0].legend(loc='lower left')
plt.savefig(plot_path+f'activationplots/jointplot_fig3_rf_kmax{kmax}_mp{150}_gammalegend.pdf')

# %%
