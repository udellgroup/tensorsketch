import numpy as np
from scipy import fftpack
import tensorly as tl
from .util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_rerr, st_hosvd
from .sketch import fetch_arm_sketch, fetch_core_sketch
import time
from tensorly.decomposition import tucker
from .recover_from_sketches import SketchTwoPassRecover
from .recover_from_sketches import SketchOnePassRecover


class TensorApprox(object):
    """
    The wrapper class for approximating the target tensor with three methods: HOOI, two-pass sketching, and one pass sketching
    """

    def __init__(self, X, ranks, ks=[], ss=[], random_seed=1, rm_typ='g', store_phis=True):
        tl.set_backend('numpy')
        self.X = X
        self.ranks = ranks
        self.ks = ks
        self.ss = ss
        self.random_seed = random_seed
        self.rm_typ = rm_typ
        self.store_phis = store_phis

    def in_memory_fix_rank_tensor_approx(self, method):
        start_time = time.time()

        if method == 'st_hosvd':
            # print(self.X.shape)
            # print(self.ranks)
            core_sketch = np.zeros(1)
            arm_sketches = [[] for _ in np.arange(len(self.X.shape))]
            tucker_core, tucker_factors = st_hosvd(self.X, self.ranks)
            X_hat = tl.tucker_to_tensor((tucker_core, tucker_factors))
            running_time = time.time() - start_time
            sketch_time = -1
            recover_time = running_time

        elif method == "hooi":
            tucker_core, tucker_factors = tucker(self.X, self.ranks, init='svd')
            X_hat = tl.tucker_to_tensor((tucker_core, tucker_factors))
            running_time = time.time() - start_time
            core_sketch = np.zeros(1)
            arm_sketches = [[] for i in np.arange(len(self.X.shape))]
            sketch_time = -1
            recover_time = running_time

        elif method == "two_pass":
            arm_sketches = fetch_arm_sketch(self.X, self.ks, typ=self.rm_typ)
            # core_sketch = fetch_core_sketch(self.X, self.ss, typ=self.rm_typ)
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, tucker_factors, tucker_core = sketch_two_pass.recover()
            recover_time = time.time() - start_time

        elif method == "one_pass":
            arm_sketches = fetch_arm_sketch(self.X, self.ks, typ=self.rm_typ)
            core_sketch, phis = fetch_core_sketch(self.X, self.ss, typ=self.rm_typ)
            # print(arm_sketches[0].shape)
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_one_pass = SketchOnePassRecover(arm_sketches, core_sketch,
                                        TensorInfoBucket(self.X.shape, self.ks, self.ranks, self.ss),
                                        RandomInfoBucket(random_seed=self.random_seed), phis)
            X_hat, tucker_factors, tucker_core = sketch_one_pass.recover()
            recover_time = time.time() - start_time
        else:
            raise Exception("please use either of the three methods: "
                            "st_hosvd, hooi, twopass, onepass")
        # Compute the the relative error when the true low rank tensor is unknown. 
        # Refer to simulation.py in case when the true low rank tensor is given. 
        rerr = eval_rerr(self.X, X_hat, self.X)
        return X_hat, (tucker_core, tucker_factors), (core_sketch, arm_sketches), rerr, (sketch_time, recover_time)


    def low_rank_tensor_approx_sketch(self, sketch, method):
        """
        :param sketch: list of two elements, first is arm_sketches (list)
        the second is the core sketch
        :param method: name for the optimization method
        :return:
        """
        # Construct the approximation directly from sketch
        start_time = time.time()
        arm_sketches, core_sketch = sketch
        if method == "hooi":
            core, tucker_factors = tucker(self.X, self.ranks, init='svd')
            X_hat = tl.tucker_to_tensor(core, tucker_factors)
            running_time = time.time() - start_time
            core_sketch = np.zeros(1)
            arm_sketches = [[] for _ in np.arange(len(self.X.shape))]
            sketch_time = -1
            recover_time = running_time

        elif method == "st_hosvd":
            core, tucker_factors = st_hosvd(self.X, self.ranks)
            X_hat = tl.tucker_to_tensor(core, tucker_factors)
            running_time = time.time() - start_time
            core_sketch = np.zeros(1)
            arm_sketches = [[] for _ in np.arange(len(self.X.shape))]
            sketch_time = -1
            recover_time = running_time

        elif method == "twopass":
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, _, _ = sketch_two_pass.recover()
            recover_time = time.time() - start_time

        elif method == "onepass":
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_one_pass = SketchOnePassRecover(arm_sketches, core_sketch, \
                                TensorInfoBucket(self.X.shape, self.ks, self.ranks, self.ss), \
                                RandomInfoBucket(random_seed=self.random_seed), \
                                sketch.get_phis())
            X_hat, _, _ = sketch_one_pass.recover()
            recover_time = time.time() - start_time
        else:
            raise Exception("please use either of the three methods: "
                            "st_hosvd, hooi, twopass, onepass")
        # Compute the the relative error when the true low rank tensor is unknown. 
        # Refer to simulation.py in case when the true low rank tensor is given. 
        rerr = eval_rerr(self.X, X_hat, self.X)
        return X_hat, core_sketch, arm_sketches, rerr, (sketch_time, recover_time)


if __name__ == "__main__":

    # Test it for square data
    n = 100
    k = 20
    rank = 5
    dim = 3
    s = 2 * k + 1
    ranks = np.repeat(rank, dim)
    ks = np.repeat(k, dim)
    ss = np.repeat(s, dim)
    tensor_shape = np.repeat(n, dim)
    noise_level = 0.1
    gen_typ = 'lk'
    X, X0 = square_tensor_gen(n, rank, dim, gen_typ, \
                              noise_level, seed=1)
    tapprox1 = TensorApprox(X, ranks, ks, ss)
    _, _, _, rerr, _ = tapprox1.tensor_approx("st_hosvd")
    print(rerr)
    _, _, _, rerr, _ = tapprox1.tensor_approx("twopass")
    print(rerr)
    _, _, _, rerr, _ = tapprox1.tensor_approx("onepass")
    print(rerr)

    # Test it for data with unequal side length

