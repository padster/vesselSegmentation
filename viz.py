from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def clean_subplots(r, c, pad=0.05, axes=False, show=False, figsize=None, dpi=None, **kwargs):
    if figsize is not None and dpi is not None:
        figsize = tuple([f / dpi for f in figsize])
    f = plt.figure(figsize=figsize, dpi=dpi)
    ax = []
    at = 1
    for i in range(r):
        row = []
        for j in range(c):
            axHere = f.add_subplot(r, c, at, **kwargs)
            if not axes:
                axHere.get_xaxis().set_visible(False)
                axHere.get_yaxis().set_visible(False)
            row.append(axHere)
            at = at + 1
        ax.append(row)
    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=pad)
    try:
        if show:
            plt.get_current_fig_manager().window.showMaximized()
    except AttributeError:
        pass # Can't maximize, sorry :(
    return f, ax
