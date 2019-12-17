import numpy as np
import pickle
import tensorsketch
from tensorsketch.tensor_approx import TensorApprox
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # CO = pickle.load(open("data/CO.pickle", 'rb'))
    T = pickle.load(open("data/T.pickle", 'rb'))
    # P = pickle.load(open("data/P.pickle", 'rb'))
    # T = np.random.normal(size = (30,30,30) )
    inv_factor = 10
    ranks = (np.array(T.shape) / inv_factor).astype(int)
    ks = (np.array(T.shape) / inv_factor).astype(int) * 2
    ss = 2 * ks + 1

    sim = tensorsketch.tensor_approx.TensorApprox(T, ranks, ks, ss, typ="g", tensor_proj = True)
    hooi_result = sim.in_memory_fix_rank_tensor_approx('hooi')
    st_hosvd_result = sim.in_memory_fix_rank_tensor_approx('st_hosvd')
    twopass_result = sim.in_memory_fix_rank_tensor_approx('two_pass')
    onepass_result = sim.in_memory_fix_rank_tensor_approx('one_pass')
    pickle.dump([hooi_result, st_hosvd_result, twopass_result, onepass_result], open('data/experiment.pickle', 'wb'))
