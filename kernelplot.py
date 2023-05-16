# %%
import numpy as np
import matplotlib.pyplot as plt

# hyperparams
gamma1=0.1
gamma2=0.01
reg=1#e-1

exec(open("myutils.py").read())

exec(open("plot_helper.py").read())


# %%
def spikysmooth_laplaces(xdiff, reg, gamma2, gamma1=1):
    return np.exp(-xdiff/ gamma1) + reg * np.exp(-xdiff / gamma2)

def spikysmooth_gaussians(xdiff, reg, gamma2, gamma1=1):
    return np.exp(-xdiff ** 2 / gamma1) + reg * np.exp(-xdiff ** 2 / gamma2)


# %%
exec(open("plot_helper.py").read())
fig, ax = plt.subplots(figsize=(1*onefigsize[0],1*onefigsize[1]))

xs2=np.linspace(-0.15,0.15,1000)
ys2=spikysmooth_laplaces(np.abs(xs2),reg=0.1,gamma2=0.01)
yslap2=spikysmooth_laplaces(np.abs(xs2),reg=0,gamma2=0.1)

ax.plot(xs2,yslap2,color='tab:blue',alpha=0.75,label='Laplace')
ax.plot(xs2,ys2,color='tab:orange',alpha=0.75,label='Spiky-smooth')
ax.annotate(r"$\{$",fontsize=48.5,alpha=0.6,
            xy=(-0.07,1.024)
            )
ax.annotate(r'$\rho$',xy=(-0.095,1.04),alpha=0.6,fontsize=24)
ax.plot([-0.04,0],[1,1],'--',color='black',alpha=0.6)
ax.plot([-0.04,0],[1.1,1.1],'--',color='black',alpha=0.6)
ax.set_yticks([0.9,1,1.1])
ax.set_xticks([-0.1,0,0.1])
ax.text(-0.0115, 0.965, s=r"$\{$", rotation=180.0*0.5, fontsize=14.5, color='black')
ax.annotate(r'$\gamma$',xy=(-0.014,0.92),fontsize=24)
ax.plot([-0.01,-0.01],[0.97,1.025],linestyle='dotted',color='black')
ax.plot([0.01,0.01],[0.97,1.025],linestyle='dotted',color='black')
#ax.grid()
plt.legend()

def correct_fontsize(axs,sizes=None):
    # use this for axs already created
    num_ax=1
    if sizes is None:
        sizes = num_ax
    for i in range(num_ax):
        for item in ([axs.title, axs.xaxis.label, axs.yaxis.label]):
            item.set_fontsize(8*sizes)
        for item in (axs.get_xticklabels() + axs.get_yticklabels()):
            item.set_fontsize(6*sizes)
        if True:
            for item in axs.legend().get_texts():
                item.set_fontsize(6*sizes)
    return axs
correct_fontsize(ax,sizes=2)
plt.savefig('outputs/spsmkernel_laplace_pure_largefonts.pdf',dpi=300)
