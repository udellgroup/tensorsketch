#######################
#                     *
#  Yiming Sun         *
#  11/2019            *
#                     *
#######################

import numpy as np
from scipy import fftpack
import tensorly as tl
from .util import square_tensor_gen, st_hosvd
from .sketch import fetch_arm_sketch, fetch_core_sketch
import time
from tensorly.decomposition import tucker
from .recover_from_sketches import SketchTwoPassRecover, SketchOnePassRecover
from .evaluate import eval_rerr



class TensorApprox(object):
    """
    The wrapper class for approximating the target tensor with three methods: HOOI, two-pass sketching, and one pass sketching
    """

    def __init__(self, X, ranks, ks=[], ss=[], **kwargs):

        tl.set_backend('numpy')
        self.X = X
        self.ranks = ranks
        self.ks = ks
        self.ss = ss
        self.random_setting = kwargs


    def in_memory_fix_rank_tensor_approx(self, method):
        start_time = time.time()

        if method == 'st_hosvd':
            # print(self.X.shape)
            # print(self.ranks)
            core_sketch = np.zeros(1)
            arm_sketches = [[] for _ in np.arange(len(self.X.shape))]
            tucker_core, tucker_factors = st_hosvd(self.X, self.ranks)
            X_hat = tl.tucker_to_tensor(tucker_core, tucker_factors)
            running_time = time.time() - start_time
            sketch_time = -1
            recover_time = running_time

        elif method == "hooi":
            core_sketch = np.zeros(1)
            arm_sketches = [[] for _ in np.arange(len(self.X.shape))]
            tucker_core, tucker_factors = tucker(self.X, self.ranks, init='svd')
            X_hat = tl.tucker_to_tensor((tucker_core, tucker_factors))
            running_time = time.time() - start_time
            sketch_time = -1
            recover_time = running_time

        elif method == "two_pass":

            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            # core_sketch = fetch_core_sketch(self.X, self.ss, typ=self.rm_typ)
            core_sketch = None
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, tucker_factors, tucker_core = \
            sketch_two_pass.in_memory_fix_rank_recover(mode='st_hosvd')
            recover_time = time.time() - start_time

        elif method == "one_pass":
            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            core_sketch, phis = fetch_core_sketch(self.X, self.ss, **self.random_setting)
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_one_pass = SketchOnePassRecover(core_sketch, arm_sketches, phis, self.ranks)
            X_hat, tucker_factors, tucker_core = sketch_one_pass.in_memory_fix_rank_recover(mode='st_hosvd')
            recover_time = time.time() - start_time
        else:
            raise Exception("please use either of the three methods: "
                            "st_hosvd, hooi, twopass, onepass")
        # Compute the the relative error when the true low rank tensor is unknown. 
        # Refer to simulation.py in case when the true low rank tensor is given. 
        rerr = eval_rerr(self.X, X_hat, self.X)
        return X_hat, (tucker_core, tucker_factors), (core_sketch, arm_sketches), rerr, (sketch_time, recover_time)


    def low_rank_tensor_approx(self, method):
        """
        :param sketch: list of two elements, first is arm_sketches (list)
        the second is the core sketch
        :param method: name for the optimization method
        :return:
        """
        # Construct the approximation directly from sketch
        start_time = time.time()

        if method == "two_pass":
            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            # core_sketch = fetch_core_sketch(self.X, self.ss, typ=self.rm_typ)
            core_sketch = None
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, tucker_factors, tucker_core = \
                sketch_two_pass.low_rank_recover()

            recover_time = time.time() - start_time

        elif method == "one_pass":
            sketch_time = time.time() - start_time
            start_time = time.time()
            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            core_sketch, phis = fetch_core_sketch(self.X, self.ss, **self.random_setting)
            sketch_one_pass = SketchOnePassRecover(core_sketch, arm_sketches, phis, self.ranks)
            X_hat, tucker_factors, tucker_core = sketch_one_pass.low_rank_recover()
            X_hat, _, _ = sketch_one_pass.low_rank_recover()
            recover_time = time.time() - start_time
        else:
            raise Exception("please use either of the three methods: "
                            "st_hosvd, hooi, twopass, onepass")
        # Compute the the relative error when the true low rank tensor is unknown. 
        # Refer to simulation.py in case when the true low rank tensor is given. 
        rerr = eval_rerr(self.X, X_hat, self.X)
        return X_hat, core_sketch, arm_sketches, rerr, (sketch_time, recover_time)


    def fix_rank_tensor_approx(self, method):
        """
        :param sketch: list of two elements, first is arm_sketches (list)
        the second is the core sketch
        :param method: name for the optimization method
        :return:
        """
        # Construct the approximation directly from sketch
        start_time = time.time()

        if method == "two_pass":
            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            # core_sketch = fetch_core_sketch(self.X, self.ss, typ=self.rm_typ)
            core_sketch = None
            sketch_time = time.time() - start_time
            start_time = time.time()
            sketch_two_pass = SketchTwoPassRecover(self.X, arm_sketches, self.ranks)
            X_hat, tucker_factors, tucker_core = \
                sketch_two_pass.fix_rank_recover()
            recover_time = time.time() - start_time

        elif method == "one_pass":
            arm_sketches, _ = fetch_arm_sketch(self.X, self.ks, **self.random_setting)
            core_sketch, phis = fetch_core_sketch(self.X, self.ss, **self.random_setting)
            sketch_time = time.time() - start_time
            sketch_one_pass = SketchOnePassRecover(core_sketch, arm_sketches, phis, self.ranks)
            start_time = time.time()
            X_hat, _, _ = sketch_one_pass.fix_rank_recover()
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

    def test_in_memory_fix_rank():
        print("now testing st_hosvd")
        _, _, _, rerr, _ = tapprox1.in_memory_fix_rank_tensor_approx("st_hosvd")
        print(rerr)

        print("now testing two pass")
        _, _, _, rerr, _ = tapprox1.in_memory_fix_rank_tensor_approx("two_pass")
        print(rerr)

        print("now testing one pass")
        _, _, _, rerr, _ = tapprox1.in_memory_fix_rank_tensor_approx("one_pass")
        print(rerr)


    def test_low_rank():

        print("now testing two pass")
        _, _, _, rerr, _ = tapprox1.low_rank_tensor_approx("two_pass")
        print(rerr)

        print("now testing one pass")
        _, _, _, rerr, _ = tapprox1.low_rank_tensor_approx("one_pass")
        print(rerr)

    def test_fix_rank():

        print("now testing two pass")
        _, _, _, rerr, _ = tapprox1.fix_rank_tensor_approx("two_pass")
        print(rerr)

        print("now testing one pass")
        _, _, _, rerr, _ = tapprox1.fix_rank_tensor_approx("one_pass")
        print(rerr)

    test_in_memory_fix_rank()
    test_fix_rank()


