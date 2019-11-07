import numpy as np
from scipy import fftpack
import tensorly as tl
import tensorsketch
from tensorsketch import util
import time
from tensorly.decomposition import tucker
from tensorsketch.tensor_approx import TensorApprox
from tensorsketch.util import square_tensor_gen


class Simulation(object):
    '''
    In this simulation, we only experiment with the square design and Gaussian 
    randomized linear map. We use the same random_seed for generating the
     data matrix and the arm matrix
    '''

    def __init__(self, n, rank, k, s, dim, gen_typ, noise_level, **kwargs):
        tl.set_backend('numpy')
        self.n, self.rank, self.k, self.s, self.dim = n, rank, k, s, dim
        self.total_num = np.prod(np.repeat(n, dim))
        self.gen_typ = gen_typ
        self.noise_level = noise_level
        self.random_setting = kwargs

    def run_sim(self):
        X, X0 = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, \
                                                    noise_level=self.noise_level)
        ranks = [self.rank for _ in range(self.dim)]
        ss = [self.s for _ in range(self.dim)]
        ks = [self.k for _ in range(self.dim)]
        tapprox = TensorApprox(X, ranks, ks=ks, ss=ss, **self.random_setting)
        X_hat_st_hosvd, _, _, _, time_st_hosvd = tapprox.in_memory_fix_rank_tensor_approx('st_hosvd')
        X_hat_twopass, _, _, _, time_twopass = tapprox.in_memory_fix_rank_tensor_approx('two_pass')
        X_hat_onepass, _, _, _, time_onepass = tapprox.in_memory_fix_rank_tensor_approx('one_pass')
        rerr_st_hosvd = tensorsketch.evaluate.eval_rerr(X, X_hat_st_hosvd, X)
        rerr_twopass = tensorsketch.evaluate.eval_rerr(X, X_hat_twopass, X)
        rerr_onepass = tensorsketch.evaluate.eval_rerr(X, X_hat_onepass, X)
        return (rerr_st_hosvd, rerr_twopass, rerr_onepass), (time_st_hosvd, time_twopass, time_onepass)


import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n = 100
    k = 10
    rank = 5
    dim = 3
    s = 2 * k + 1
    tensor_shape = np.repeat(n, dim)
    noise_level = 0.01
    gen_typ = 'lk'

    noise_levels = (np.float(10) ** (np.arange(-10, 2, 2)))
    hooi_rerr = np.zeros(len(noise_levels))
    two_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr_ns = np.zeros(len(noise_levels))

    for idx, noise_level in enumerate(noise_levels):
        print('Noise_level:', noise_level)
        simu = Simulation(n, rank, k, s, dim, gen_typ, noise_level)
        (rerr_hooi, rerr_twopass, rerr_onepass), _ = simu.run_sim()
        # print('hooi rerr:', rerr)
        hooi_rerr[idx] = rerr_hooi
        two_pass_rerr[idx] = rerr_twopass
        one_pass_rerr[idx] = rerr_onepass

    print("identity design with varying noise_level")
    print("noise_levels", noise_levels)
    print("hooi", hooi_rerr)
    print("two_pass", two_pass_rerr)
    print("one_pass", one_pass_rerr)
    print("one_pass_ns", one_pass_rerr_ns)

    plt.subplot(3, 1, 1)
    plt.plot(noise_levels, hooi_rerr, label='st_hosvd')
    plt.title('st_hosvd')
    plt.subplot(3, 1, 2)
    plt.plot(noise_levels, two_pass_rerr, label='two_pass')
    plt.title('two_pass')
    plt.subplot(3, 1, 3)
    plt.plot(noise_levels, one_pass_rerr, label='one_pass')
    plt.title('one_pass')
    plt.show()
