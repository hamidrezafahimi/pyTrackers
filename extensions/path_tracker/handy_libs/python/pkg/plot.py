import matplotlib.pyplot as plt

def plot_now(plot=None, scatter=None, pause=None, title=None, xl=None, yl=None):
    plt.cla()
    if not plot is None:
        for p in plot:
            if len(p) == 2:
                plt.plot(p[0], p[1])
            elif len(p) == 3:
                plt.plot(p[0], p[1], label=p[2])
    if not scatter is None:
        for p in scatter:
            if len(p) == 2:
                plt.scatter(p[0], p[1])
            elif len(p) == 3:
                plt.scatter(p[0], p[1], label=p[2])
    plt.grid(True)
    if not title is None:
        plt.title(title)
    plt.legend()
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)