# %%
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

mp.dps = 50

# hyperparams
gamma1=0.1
gamma2=0.01
reg=1#e-1
kmax=1000
kmaxs=np.array([5,10,20,30,50])
#gamma2s=np.array([0.0005,0.001,0.005,0.01,0.05,0.1])
gamma2s=np.array(mp.matrix(np.array([0.02,0.04,0.07,0.2,0.5])))
n_figs=len(gamma2s)
xs = np.array(mp.matrix(np.linspace(-3,3,200)))
lights= np.linspace(0.5,2,len(gamma2s))
num_rand=3
n_figs=len(gamma2s)

exec(open("myutils.py").read())
exec(open("plot_helper.py").read())
# loaded these:
plot_path, find, mysave, myload, onefigsize = plot_path, find, mysave, myload, onefigsize

# %%
# use this for plotting spiky-smooth activation functions (Appendix)
# first, find the precomputed file containing the activation function evaluations
# plot for different choices of signs s_i
n_figs=4
sizes=2
for ker in ['ntk','rf']:
    for filenam in find(f'activation_{ker}_spikysmooth*mpmath150_*kmax{kmax}_x3*smallgamma.txt','outputs/'):
        gamma2s,xs,activs1,activsppmm,activsrand,pmsrand = myload(filenam)
        
        fig,ax = plt.subplots(3,n_figs, figsize=(n_figs*onefigsize[0],3*onefigsize[1]))
        for i,gamma2 in enumerate(gamma2s[:-1]):
            ax[0,n_figs-1 - i].plot(xs,activs1[i])
            ax[1,n_figs-1 - i].plot(xs,activsppmm[i])

            for _ in range(num_rand):
                ax[2,n_figs-1 - i].plot(xs,activsrand[int(3*i+_)])
            ax[2,n_figs-1-i].set_xlabel(r'$\mathbf{\gamma} =$'+f' {gamma2}',fontweight='bold')

        ax[0,0].set_ylabel('+++...')
        ax[1,0].set_ylabel('+ + - -...')
        ax[2,0].set_ylabel('Random +-')
        for i in range(n_figs):
            for j in range(3):
                ax[j,i].set_xlim(-3.01,3.01)
                ax[j,i].plot([-4,4],[0,0],color = 'black',linewidth =1,alpha=0.4,zorder=-10)
                ax[j,i].plot([0,0],ax[j,i].get_ylim(),color = 'black',linewidth =1,alpha=0.4,zorder=-10)
                for item in ([ax[j,i].xaxis.label, ax[j,i].yaxis.label]):
                    item.set_fontsize(8*sizes)
                for item in (ax[j,i].get_xticklabels() + ax[j,i].get_yticklabels()):
                    item.set_fontsize(6*sizes)
        plt.savefig(plot_path+'activationplots/jointplot_'+filenam[:-4].split('/',10)[-1]+'_coordax4inrow.pdf')

# %%
