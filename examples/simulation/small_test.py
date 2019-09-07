import numpy as np
from tensorsketch.tensor_approx import TensorApprox
from tensorsketch.util import square_tensor_gen
import tensorly as tl

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


ranks = np.array((5, 10, 15))
dim = 3
ns = np.array((100, 200, 300))
ks = np.array((15, 20, 25))
ss = 2 * ks + 1
core_tensor = np.random.uniform(0, 1, ranks)
arms = []
tensor = core_tensor
for i in np.arange(dim):
    arm = np.random.normal(0, 1, size=(ns[i], ranks[i]))
    arm, _ = np.linalg.qr(arm)
    arms.append(arm)
    tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
true_signal_mag = np.linalg.norm(core_tensor) ** 2
noise = np.random.normal(0, 1, ns)
X = tensor + noise * np.sqrt((noise_level ** 2) * true_signal_mag / np.product \
(np.prod(ns)))
tapprox2 = TensorApprox(X, ranks, ks, ss)
_, _, _, rerr, _ = tapprox2.tensor_approx("st_hosvd")
print(rerr)
_, _, _, rerr, _ = tapprox2.tensor_approx("twopass")
print(rerr)
_, _, _, rerr, _ = tapprox2.tensor_approx("onepass")
print(rerr)

