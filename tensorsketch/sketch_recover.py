import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from .sketch import fetch_arm_sketch, fetch_core_sketch
from .util import square_tensor_gen, eval_rerr, st_hosvd
from scipy.sparse.linalg import svds


class SketchTwoPassRecover(object):

    def __init__(self, X, arm_sketches, ranks):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
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
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q[:self.rank[i]])

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

    def __init__(self, arm_sketches, core_sketch, Tinfo_bucket, Rinfo_bucket, \
                 phis=[], rm_typ="g"):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.arm_sketches = arm_sketches
        # Note get_info extract some extraneous information
        self.tensor_shape, self.ks, self.ranks, self.ss = Tinfo_bucket.get_info()
        self.Rinfo_bucket = Rinfo_bucket
        self.phis = phis
        self.core_sketch = core_sketch
        self.rm_typ = rm_typ

    def get_phis(self):
        '''
        Obtain phis from the sketch when phis is not stored
        '''
        phis = []
        rm_generator = Sketch.sketch_core_rm_generator(self.tensor_shape, self.ss, \
                                                       self.Rinfo_bucket)
        for rm in rm_generator:
            phis.append(rm)
        return phis

    def recover(self, mode='st_hosvd'):
        '''
        Obtain the recovered tensor X_hat, core and arm tensor given the sketches
        using the one pass sketching algorithm 
        '''
        if self.phis == []:
            phis = self.get_phis()
        else:
            phis = self.phis
        Qs = []
        for arm_sketch in self.arm_sketches:
            Q, _ = np.linalg.qr(arm_sketch)
            Qs.append(Q)
        self.core_tensor = self.core_sketch
        dim = len(self.tensor_shape)
        for mode_n in range(dim):
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, \
                                                  np.linalg.pinv(np.dot(phis[mode_n], Qs[mode_n])), mode=mode_n)
        if mode == 'hooi':
            self.core_tensor, factors = tucker(self.core_tensor, ranks=self.ranks)

        elif mode == 'st_hosvd':
            self.core_tensor, factors = st_hosvd(self.core_tensor, target_ranks=self.ranks)

        for n in range(dim):
            self.arms.append(np.dot(Qs[n], factors[n]))

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