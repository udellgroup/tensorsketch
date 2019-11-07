import tensorly as tl
import numpy as np
from .util import random_matrix_generator, square_tensor_gen
from .util import ssrft_modeprod, gprod, sp0prod
from .random_projection import random_matrix_generator, tensor_random_matrix_generator
from sklearn.decomposition import TruncatedSVD


def fetch_arm_sketch(X, ks, tensor_proj=True, **kwargs_rg):
    """
    :param X: the tensor of dimension N
    :param ks: array of size N
    :param tensor_proj: True: use tensor random projection,
    otherwise, use normal one
    :param kwargs_rg:
    :return: list of two, first element is list of arm sketches
    and the second one is list of random matrices with size I_n\times k_n
    """
    arm_sketches = []
    omegas = []
    for i, n in enumerate(X.shape):
        shape = list(X.shape)
        del shape[i]
        if not tensor_proj:
            omega = random_matrix_generator(shape, ks[i], **kwargs_rg)
            arm_sketch = tl.unfold(X, mode=i) @ omega
            arm_sketches.append(arm_sketch)
            omegas.append(omega)
        else:
            omega = tensor_random_matrix_generator(shape, ks[i], **kwargs_rg)
            arm_sketch = tl.unfold(X, mode=i) @ omega
            arm_sketches.append(arm_sketch)
            omegas.append(omega)
    return arm_sketches, omegas



def fetch_core_sketch(X, ss, **kwargs_rg):
    """
    :param X: the tensor of dimension N
    :param ks: array of size N
    :param tensor_proj: True: use tensor random projection,
    otherwise, use normal one
    :return: [core_sketch:s_n\times s_n ...\times s_n,
    list of sketches phis, s_n\times I_n]
    """
    core_sketch = X
    phis = []
    shape = list(core_sketch.shape)
    for mode, n in enumerate(X.shape):
        phi = random_matrix_generator(shape[mode], ss[mode], **kwargs_rg)
        phis.append(phi)
        core_sketch = tl.tenalg.mode_dot(core_sketch, phi.transpose(), mode = mode)
    return core_sketch,  phis



if __name__ == "__main__":

    tl.set_backend('numpy')

    def test_arm_sketches():
        X, X0 = square_tensor_gen(50, 3, dim=3, typ='spd', noise_level=0.1)
        arms, omegas = fetch_arm_sketch(X, [10, 10, 10], tensor_proj=True, typ='g')
        print(f"length of arms is {len(arms)} and length of omega is {len(omegas)}")
        print(arms[0].shape)
        print(omegas[0].shape)

    test_arm_sketches()

    def test_core_sketch():
        X, X0 = square_tensor_gen(50, 3, dim=3, typ='spd', noise_level=0.1)
        core_sketch, phis = fetch_core_sketch(X, [21, 21, 21], typ='g')
        print(core_sketch.shape)

    test_core_sketch()


