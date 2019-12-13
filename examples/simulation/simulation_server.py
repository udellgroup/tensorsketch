# coding: utf-8

# In[1]:


import numpy as np
from scipy import fftpack
import tensorly as tl
import time
from tensorly.decomposition import tucker
import tensorsketch
from tensorsketch import util
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import simulation
import warnings

warnings.filterwarnings('ignore')


class ClassName(object):
    """docstring for ClassName"""

    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


# In[2]:


def sim_name(gen_type, r, noise_level, dim, typ, tensor_proj):
    """
    Obtain the file name to use for a given simulation setting
    """
    if noise_level == 0:
        noise = "no"
    else:
        noise = str(int(np.log10(noise_level)))
    if tensor_proj: 
        t = "prod"
    else:
        t = ""
    return "data/typ" + gen_type + "_r" + str(r) + "_noise" + noise + "_dim" + str(dim) + "_" + typ +t

# In[3]:


def run_nssim(gen_type, r, noise_level, ns=np.arange(100, 101, 100), dim=3, sim_runs=1, random_seed=1, **kwargs):
    """
    Simulate multiple datasets with different n for multiple runs. For each run, perform the HOOI, 
    two pass sketching, and one pass sketching 
    
    :param gen_type: Type of random matrix used in sketching, including 'u' uniform, 'g' gaussian, 'sp'
        sparse radamacer. 
    :param r: tucker rank of the simulated tensor 
    :param noise_level: noise level. It inverse equals to the signal-to-noise ratio.  
    :param ns: array of different n, the side length of the square tensor 
    :param dim: the dimension of the square tensor 
    :param sim_runs: num of simulated runs in each setting 
    :param random_seed: random seed for generating the random matrix  
    """
    np.random.seed(random_seed)
    sim_list = []
    sim_time_list = [[], []]
    for id, n in enumerate(ns):
        if gen_type in ['id', 'lk']:
            ks = np.arange(r+1, int(n / 2), int(n / 20))
        elif gen_type in ['spd', 'fpd']:
            ks = np.arange(r+1, int(n / 5), int(n / 50))
        else:
            ks = np.arange(r+1, int(n / 10), int(n / 100))
        hooi_rerr = np.zeros((sim_runs, len(ks)))
        st_hosvd_rerr = np.zeros((sim_runs, len(ks)))
        two_pass_rerr = np.zeros((sim_runs, len(ks)))
        one_pass_rerr = np.zeros((sim_runs, len(ks)))

        hooi_sketch_time = np.zeros((sim_runs, len(ks)))
        st_hosvd_sketch_time = np.zeros((sim_runs, len(ks)))
        two_pass_sketch_time = np.zeros((sim_runs, len(ks)))
        one_pass_sketch_time = np.zeros((sim_runs, len(ks)))

        hooi_recover_time = np.zeros((sim_runs, len(ks)))
        st_hosvd_recover_time = np.zeros((sim_runs, len(ks)))
        two_pass_recover_time = np.zeros((sim_runs, len(ks)))
        one_pass_recover_time = np.zeros((sim_runs, len(ks)))

        for i in range(sim_runs):
            for idx, k in enumerate(ks):
                simu = simulation.Simulation(n, r, k, 2 * k + 1, dim, gen_type,
                                             noise_level, **kwargs)
                (rerr_hooi, rerr_st_hosvd, rerr_twopass, rerr_onepass), (time_hooi, time_st_hosvd, time_twopass, time_onepass) = simu.run_sim()
                hooi_rerr[i, idx] = rerr_hooi
                st_hosvd_rerr[i, idx] = rerr_st_hosvd
                two_pass_rerr[i, idx] = rerr_twopass
                one_pass_rerr[i, idx] = rerr_onepass

                hooi_sketch_time[i, idx] = time_hooi[0]
                st_hosvd_sketch_time[i, idx] = time_st_hosvd[0]
                two_pass_sketch_time[i, idx] = time_twopass[0]
                one_pass_sketch_time[i, idx] = time_onepass[0]

                hooi_recover_time[i, idx] = time_hooi[1]
                st_hosvd_recover_time[i, idx] = time_st_hosvd[1]
                two_pass_recover_time[i, idx] = time_twopass[1]
                one_pass_recover_time[i, idx] = time_onepass[1]

        sim_list.append([two_pass_rerr, one_pass_rerr, hooi_rerr, st_hosvd_rerr])
        sim_time_list[0].append([two_pass_sketch_time, one_pass_sketch_time, hooi_sketch_time, st_hosvd_sketch_time])
        sim_time_list[1].append([two_pass_recover_time, one_pass_recover_time, hooi_recover_time, st_hosvd_recover_time])

    pickle.dump(sim_list, open(sim_name(gen_type, r, noise_level, dim, typ, tensor_proj) + ".pickle", "wb"))
    pickle.dump(sim_time_list, open(sim_name(gen_type, r, noise_level, dim, typ, tensor_proj) + "_time.pickle", "wb"))
    return sim_list


# In[4]:


def plot_nssim(gen_type, r, noise_level, name, n, ns=[200, 400, 600], dim=3, sim_runs=1, random_seed=1, fontsize=18,
               rm_typ="g"):
    '''
    Plot the simulation results given in run_nssim
    '''
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    sim_list = pickle.load(open(sim_name(gen_type, r, noise_level, dim, rm_typ) + ".pickle", "rb"))
    plot_id = ns.index(n)
    if gen_type in ['id', 'lk']:
        ks = np.arange(r, int(n / 2), int(n / 20))
    elif gen_type in ['spd', 'fpd']:
        ks = np.arange(r, int(n / 5), int(n / 50))
    else:
        ks = np.arange(r, int(n / 10), int(n / 100))
    plt.figure(figsize=(6, 5))
    plt.plot(ks / n, np.mean(sim_list[plot_id][0], 0), label='Two Pass', markersize=10, linestyle='--', marker='X')
    plt.plot(ks / n, np.mean(sim_list[plot_id][1], 0), label='One Pass', markersize=10, marker='s', markeredgewidth=1,
             markeredgecolor='orange', markerfacecolor='None')
    plt.plot(ks / n, np.mean(sim_list[plot_id][2], 0), label='HOOI', markersize=10, linestyle=':', marker='o',
             markeredgewidth=1, markeredgecolor='g', markerfacecolor='None')
    plt.title("I = %s" % (n))
    plt.legend(loc='best')
    plt.xlabel('Compression Factor: $\delta_1$ = k/I')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.minorticks_off()
    alldata = np.concatenate(
        (np.mean(sim_list[plot_id][0], 0), np.mean(sim_list[plot_id][1], 0), np.mean(sim_list[plot_id][2], 0)))
    ymin = min(alldata)
    ymax = max(alldata)

    def round_to_n(x, n):
        return round(x, -int(np.floor(np.log10(abs(x)))) + n - 1)

    ticks = [round_to_n(i, 3) for i in np.arange(ymin, ymax + (ymax - ymin) / 5, (ymax - ymin) / 5)]
    plt.yticks(ticks)
    plt.axes().title.set_fontsize(fontsize)
    plt.axes().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.axes().xaxis.label.set_fontsize(fontsize)
    plt.axes().yaxis.label.set_fontsize(fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig('plots/' + name)
    plt.show()





if __name__ == '__main__':
    run_nssim('lk', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('lk', 5, 0.1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('lk', 5, 1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('spd', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('fpd', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('fed', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('sed', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('slk', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('slk', 5, 0.1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('slk', 5, 1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('id', 5, 0.01, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('id', 5, 0.1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)
    run_nssim('id', 5, 1, np.arange(200, 601, 200), typ="sp0", tensor_proj = True)

    run_nssim('lk', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('lk', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('lk', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('spd', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('fpd', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('fed', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('sed', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('slk', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('slk', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('slk', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('id', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('id', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = False)
    run_nssim('id', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = False)

    run_nssim('lk', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('lk', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('lk', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('spd', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('fpd', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('fed', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('sed', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('slk', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('slk', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('slk', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('id', 5, 0.01, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('id', 5, 0.1, np.arange(200, 601, 200), typ="g", tensor_proj = True)
    run_nssim('id', 5, 1, np.arange(200, 601, 200), typ="g", tensor_proj = True)

    run_nssim('lk', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('lk', 5, 0.1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('lk', 5, 1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('spd', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('fpd', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('fed', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('sed', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('slk', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('slk', 5, 0.1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('slk', 5, 1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('id', 5, 0.01, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('id', 5, 0.1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)
    run_nssim('id', 5, 1, np.arange(200, 601, 200), typ="ssrft", tensor_proj = False)

''' 
    run_nssim_fk('id',5,0.01,600, rm_typ = "ssrft")
    run_nssim_fk('id',5,0.1,600, rm_typ = "ssrft")
    run_nssim_fk('id',5,1,600, rm_typ = "ssrft")
    run_nssim_fk('lk',5,0.01,600, rm_typ = "ssrft")
    run_nssim_fk('lk',5,0.1,600, rm_typ = "ssrft")
    run_nssim_fk('lk',5,1,600, rm_typ = "ssrft")
    run_nssim_fk('spd',5,0.01,600, rm_typ = "ssrft")
    run_nssim_fk('fpd',5,0.01,600, rm_typ = "ssrft")
    run_nssim_fk('sed',5,0.01,600, rm_typ = "ssrft")
    run_nssim_fk('fed',5,0.01,600, rm_typ = "ssrft")

    run_nssim_fk('id',5,0.01,600, rm_typ = "u")
    run_nssim_fk('id',5,0.1,600, rm_typ = "u")
    run_nssim_fk('id',5,1,600, rm_typ = "u")
    run_nssim_fk('lk',5,0.01,600, rm_typ = "u")
    run_nssim_fk('lk',5,0.1,600, rm_typ = "u")
    run_nssim_fk('lk',5,1,600, rm_typ = "u")
    run_nssim_fk('spd',5,0.01,600, rm_typ = "u")
    run_nssim_fk('fpd',5,0.01,600, rm_typ = "u")
    run_nssim_fk('sed',5,0.01,600, rm_typ = "u")
    run_nssim_fk('fed',5,0.01,600, rm_typ = "u")

    run_nssim_fk('id',5,0.01,600, rm_typ = "gprod")
    run_nssim_fk('id',5,0.1,600, rm_typ = "gprod")
    run_nssim_fk('id',5,1,600, rm_typ = "gprod")
    run_nssim_fk('lk',5,0.01,600, rm_typ = "gprod")
    run_nssim_fk('lk',5,0.1,600, rm_typ = "gprod")
    run_nssim_fk('lk',5,1,600, rm_typ = "gprod")
    run_nssim_fk('spd',5,0.01,600, rm_typ = "gprod")
    run_nssim_fk('fpd',5,0.01,600, rm_typ = "gprod")
    run_nssim_fk('sed',5,0.01,600, rm_typ = "gprod")
    run_nssim_fk('fed',5,0.01,600, rm_typ = "gprod")

'''
