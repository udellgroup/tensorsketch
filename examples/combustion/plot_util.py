import matplotlib.pyplot as plt
import numpy as np 
MARKER_LIST = ["s", "x", "o", "+", "*", "d", "^", "v"]
MARKER_COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lawngreen', 'violet']


def find_rm_label(rm_typ):
    if rm_typ == "g":
        return "Gaussian"
    elif rm_typ == "u":
        return "Uniform"
    elif rm_typ == "sp0":
        return "Sparse"
    elif rm_typ == "gprod":
        return "Gaussian TRP"
    elif rm_typ == "ssrft":
        return "SSRFT"
    elif rm_typ == "sp0prod":
        return "Sparse TRP"


def find_gen_label(gen_typ):
    if gen_typ == "id":
        return "Superdiagonal"
    elif gen_typ == "lk":
        return "Low Rank"
    elif gen_typ == "fed":
        return "Fast Exponential Decay"
    elif gen_typ == "sed":
        return "Slow Exponential Decay"
    elif gen_typ == "fpd":
        return "Fast Polynomial Decay"
    elif gen_typ == "spd":
        return "Polynomial Decay"
    elif gen_typ == "slk":
        return "Sparse Low Rank"


def set_plot(fontsize):
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.axes().title.set_fontsize(fontsize)
    plt.axes().xaxis.label.set_fontsize(fontsize)
    plt.axes().yaxis.label.set_fontsize(fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.tight_layout()


def marker_color(method, rm_typ='gprod'):
    pairs = [('TS', 'gprod'), ('2pass', 'gprod'), ('1pass', 'gprod'), ('1pass', 'sp0prod'), ('1pass', 'ssrft'),
             ('1pass', 'gprod'), ('2pass', 'gprod'), ('1pass', 'g')]
    if (method, rm_typ) in pairs:
        return MARKER_COLOR_LIST[pairs.index((method, rm_typ))]
    else:
        return np.random.rand(3)


def marker(method, rm_typ='gprod'):
    pairs = [('TS', 'gprod'), ('2pass', 'gprod'), ('1pass', 'gprod'), ('1pass', 'sp0prod'), ('1pass', 'ssrft'),
             ('1pass', 'gprod'), ('2pass', 'gprod'), ('1pass', 'g')]
    if (method, rm_typ) in pairs:
        return MARKER_LIST[pairs.index((method, rm_typ))]
    else:
        return 's'
