#######################
#                     *
#  Yiming Sun         *
#  2019               *
#                     *
#######################

"""
This file contains class
to return an approximation from
sketches. Two pass algorithm also need the original
tensor while one pass algorithm requires
"""




import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from .sketch import fetch_arm_sketch, fetch_core_sketch
from .util import square_tensor_gen, eval_rerr, st_hosvd
from scipy.sparse.linalg import svds

def check_sketch_size_valid(arm_sketches, core_sketch, ranks, typ='i'):
    """
    :param arm_sketches:  arm_sketch (I_n k_n)
    :param core_sketch:  (s_1, ..., s_n)
    :param ranks: (r_1, ..., r_n)
    :param typ: 'i' means in memory, 'f' fix rank
    :return:
    """
    ss = core_sketch.shape
    kk = [arm_sketch.shape[1] for arm_sketch in arm_sketches]
    flag = []
    for i, s in enumerate(ss):
        if typ == 'i'
            flag.append((s > ranks[i]) & (s>kk[i]) & (kk[i]>ranks[i]))
        elif typ == 'f':
            flag.append((s > ranks[i]) & (kk[i] > ranks[i]))
        else:
            raise Exception('typ is not supported')
    assert all(flag), "Correct the size for sketches you input"



class SketchTwoPassRecover(object):
    """
    return a low rank approximation from sketches(arm sketches only here)
    """

    def __init__(self, X, arm_sketches, ranks):
        tl.set_backend('numpy')
        self.arms = []
        self.X = X
        self.arm_sketches = arm_sketches
        self.ranks = ranks

    def in_memory_fix_rank_recover(self, mode='st_hosvd'):
        '''
        Obtain the recovered tensor X_hat, core and arm tensor given the sketches
        using the two pass sketching algorithm 
        '''
        # get orthogonal basis for each arm
        Qs = []
        for sketch in self.arm_sketches:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)

        # get the core_(smaller) to implement tucker
        self.core_tensor = self.X
        N = len(self.X.shape)
        for mode_n in range(N):
            Q = Qs[mode_n]
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, Q.T, mode=mode_n)
        if mode == 'hooi':
            self.core_tensor, factors = tucker(self.core_tensor, ranks=self.ranks)
        elif mode == 'st_hosvd':
            self.core_tensor, factors = st_hosvd(self.core_tensor, target_ranks=self.ranks)


        # arm[n] = Q.T*factors[n]
        for n in range(len(factors)):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = tl.tucker_to_tensor((self.core_tensor, self.arms))
        return X_hat, self.arms, self.core_tensor

    def fix_rank_recover(self):
        '''
        Obtain the recovered tensor X_hat, core and arm tensor given the sketches
        using the two pass sketching algorithm
        '''
        # get orthogonal basis for each arm

        Qs = []
        for i, sketch in enumerate(self.arm_sketches):
            # truncated svd
            u, _, _ = svds(sketch, self.ranks[i])
            Qs.append(u)

        # get the core_(smaller) to implement tucker
        self.core_tensor = self.X
        N = len(self.X.shape)
        for mode_n in range(N):
            Q = Qs[mode_n]
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, Q.T, mode=mode_n)

        self.arms = Qs
        X_hat = tl.tucker_to_tensor((self.core_tensor, self.arms))
        return X_hat, self.arms, self.core_tensor




class SketchOnePassRecover(object):


    def __init__(self, X, arm_sketches, core_sketch, phis, ranks):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.arm_sketches = arm_sketches
        # Note get_info extract some extraneous information
        self.phis = phis
        self.core_sketch = core_sketch
        self.ranks = ranks



    def in_memory_fix_rank_recover(self, mode='st_hosvd'):
        '''
        Obtain the recovered tensor X_hat, core and arm tensor given the sketches
        using the one pass sketching algorithm 
        '''
        check_sketch_size_valid(self.arm_sketches, self.core_sketch, self.ranks, typ='i')

        Qs = []
        for arm_sketch in self.arm_sketches:
            Q, _ = np.linalg.qr(arm_sketch)
            Qs.append(Q)
        self.core_tensor = self.core_sketch
        dim = len(self.X.shape)
        for mode_n in range(dim):
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, \
                                                  np.linalg.pinv(np.dot(self.phis[mode_n], Qs[mode_n])), mode=mode_n)
        if mode == 'hooi':
            self.core_tensor, factors = tucker(self.core_tensor, ranks=self.ranks)

        elif mode == 'st_hosvd':
            self.core_tensor, factors = st_hosvd(self.core_tensor, target_ranks=self.ranks)

        for n in range(dim):
            self.arms.append(np.dot(Qs[n], factors[n]))

        X_hat = tl.tucker_to_tensor((self.core_tensor, self.arms))

        return X_hat, self.arms, self.core_tensor

    def fix_rank_recover(self):

        check_sketch_size_valid(self.arm_sketches, self.core_sketch, self.ranks, typ='f')
        Qs = []
        for i, sketch in enumerate(self.arm_sketches):
            # truncated svd
            u, _, _ = svds(sketch, self.ranks[i])
            Qs.append(u)
        self.core_tensor = self.core_sketch
        dim = len(self.X.shape)
        for mode_n in range(dim):
            u, _, _ = svds(sketch, self.ranks[i])
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, \
                            np.linalg.pinv(np.dot(self.phis[mode_n].transpose(), Qs[mode_n])), mode=mode_n)

        X_hat = tl.tucker_to_tensor((self.core_tensor, self.arms))
        return X_hat, self.arms, self.core_tensor



def test_st_hosvd():
    X, X0 = square_tensor_gen(20, 3, dim=3, typ='lk', noise_level=0.1, seed=None, sparse_factor=0.2)
    core, arms = st_hosvd(X, 3)
    print(core.shape, arms[0].shape)
    X_hat = tl.tucker_to_tensor((core, arms))
    print(eval_rerr(X, X_hat, X))


if  __name__ == "__main__":
    test_st_hosvd()