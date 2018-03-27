import matplotlib.pyplot as plt

def clean_subplots(r, c, pad=0.05, show=True):
    f = plt.figure()
    ax = []
    at = 1
    for i in range(r):
        row = []
        for j in range(c):
            axHere = f.add_subplot(r, c, at)
            axHere.get_xaxis().set_visible(False)
            axHere.get_yaxis().set_visible(False)
            row.append(axHere)
            at = at + 1
        ax.append(row)
    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=pad)
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except AttributeError:
        pass # Can't maximize, sorry :(
    return ax
