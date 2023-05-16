# %%
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']


plot_path = 'plots/'

import string
def enumerate_subplots(axs, pos_x=-0.08, pos_y=1.05, fontsize=16):
    """Adds letters to subplots of a figure.
    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.
    Returns:
        axs (list): List of plt.axes.
    """
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            pos_x[n],
            pos_y[n],
            f"{string.ascii_lowercase[n]}.",
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
        )
    plt.tight_layout()
    return axs

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_fontsize(num_cols):
    # use this before creating the axes
    keys = ['font.size','axes.labelsize','legend.fontsize','xtick.labelsize','ytick.labelsize','axes.titlesize']
    for key in keys:
        plt.rcParams.update({key: bundles.icml2022()[key] * 2 / 2})

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
    
def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


adjust_fontsize(3)

# %%
# if not os.path.exists(plot_path+'activationplots/'):
#     os.mkdir(plot_path+'activationplots/')
