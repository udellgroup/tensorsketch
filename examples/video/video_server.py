import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorly as tl
import matplotlib.ticker as ticker
import tensorsketch
from tensorsketch.tensor_approx import TensorApprox, eval_rerr
import warnings
from tensorly.decomposition import tucker
from tensorsketch.util import RandomInfoBucket, random_matrix_generator
import time

warnings.filterwarnings('ignore')


def update_sketch(arm_sketches, core_sketch, data, ks, ss, t_dim=0, t_start=0, t_end=0, rm_typ='g', typ='turnstile',
                  seed=1):
    if typ == 'turnstile':
        new_sketch = tensorsketch.sketch.Sketch(data, ks, seed, ss)
        [new_arm_sketches, new_core_sketch] = new_sketch.get_sketches()
        for i in range(len(arm_sketches)):
            arm_sketches[i] = arm_sketches[i] + new_arm_sketches[i]
        core_sketch = core_sketch + new_core_sketch
        # print(new_core_sketch)
        # print(new_arm_sketches)
        return [arm_sketches, core_sketch]
    elif typ == "turnstile_ts":
        pass


def init_sketch(data_dim, ks, ss):
    arm_sketches = []
    for i in np.arange(len(data_dim)):
        arm_sketches.append(np.zeros([data_dim[i], ks[i]]))
    core_sketch = np.zeros(ss)
    return [arm_sketches, core_sketch]


def video_sketch(data_files, data_dim=[2493, 1080, 1920], ks=[20, 20, 20], ss=[41, 41, 41], rm_typ='gprod',
                 typ='turnstile', seed=1):
    [arm_sketches, core_sketch] = init_sketch(data_dim, ks, ss)
    for idx, data_file in enumerate(data_files):
        print(idx)
        data = np.load(data_file)
        data_aug = np.zeros(data_dim)
        data_aug[idx * int(data_dim[0] / len(data_files)):(idx + 1) * int(data_dim[0] / len(data_files)), :, :] = data
        [arm_sketches, core_sketch] = update_sketch(arm_sketches, core_sketch, data_aug, ks, ss, rm_typ=rm_typ, typ=typ,
                                                    seed=1)
    return [arm_sketches, core_sketch]


def store_sketch(sketch, name='walk'):
    arm_sketches, core_sketch = sketch
    for idx, arm_sketch in enumerate(arm_sketches):
        np.save("data/" + name + "_arm" + str(idx) + ".npy", arm_sketch)
    np.save("data/" + name + "_core.npy", core_sketch)


def load_sketch(name, dim):
    arm_sketches = []
    for i in np.arange(dim):
        # arm_sketches.append(np.load("/data/yg93/video/"+name+"_arm"+str(i)+".npy"))
        arm_sketches.append(np.load("data/" + name + "_arm" + str(i) + ".npy"))
    # core_sketch = np.load("/data/yg93/video/"+name+"_core.npy")
    core_sketch = np.load("data/" + name + "_core.npy")
    return [arm_sketches, core_sketch]


def twopass_video_recover(arm_sketches, ranks):
    '''
    Two pass recover in a streaming way along the first dimension
    '''
    # get orthogonal basis for each arm
    arms = []
    Qs = []
    for sketch in arm_sketches:
        Q, _ = np.linalg.qr(sketch)
        Qs.append(Q)

    N = len(arm_sketches)
    cores = []
    # get the core_(smaller) to implement tucker
    for i in range(9):
        # Second pass of the data
        # X = np.load("/data/yg93/video/grey_walk"+str(i)+".npy")
        X = np.load("data/grey_walk" + str(i) + ".npy")
        core = X
        for mode_n in range(1, N):
            Q = Qs[mode_n]
            core = tl.tenalg.mode_dot(core, Q.T, mode=mode_n)
        cores.append(core)
    core_tensor = np.concatenate(cores, axis=0)
    core_tensor = tl.tenalg.mode_dot(core_tensor, Qs[0].T, mode=0)
    core_tensor, factors = tucker(core_tensor, ranks=ranks)

    # arm[n] = Q.T*factors[n]
    for n in range(len(factors)):
        arms.append(np.dot(Qs[n], factors[n]))
    X_hat = tl.tucker_to_tensor(core_tensor, arms)
    return X_hat, arms, core_tensor


if __name__ == '__main__':
    # for k in np.array([300]):
    #     print('One Pass')
    #     print(k)
    #     ks = np.repeat(k,3)
    #     ss = 2*ks+1
    #     time0 = time.time()
    #     walk_sketch = load_sketch('walk_k'+str(k),3) 
    #     load_time = time.time() - time0
    #     print("Load time: "+str(load_time))
    #     arm_sketches, core_sketch = walk_sketch 
    #     r = 50
    #     time1 = time.time()
    #     sketch_one_pass = tensorsketch.sketch_recover.SketchOnePassRecover(arm_sketches, core_sketch, \
    #                 tensorsketch.util.TensorInfoBucket([2493,1080,1920], ks, np.array([50, 50, 3]) , ss),\
    #                 tensorsketch.util.RandomInfoBucket(random_seed = 1)) 
    #     _ ,tucker_arms, tucker_core = sketch_one_pass.recover()
    #     recover_time = time.time() - time1
    #     print('Recover time: '+ str(recover_time))
    #     pickle.dump([tucker_arms,tucker_core, recover_time], open("data/walk_tucker_k"+str(k)+"_r"+str(50_50_3)+".pickle",'wb'))

    for k in np.array([300]):
        print('Two Pass')
        print(k)
        ks = np.repeat(k, 3)
        ss = 2 * ks + 1
        time0 = time.time()
        walk_sketch = load_sketch('walk_k' + str(k), 3)
        load_time = time.time() - time0
        print("Load time: " + str(load_time))

        time1 = time.time()
        arm_sketches, core_sketch = walk_sketch
        r = 50
        _, tucker_arms, tucker_core = twopass_video_recover(arm_sketches, np.array([50, 50, 3]))
        recover_time = time.time() - time1
        print('Recover time: ' + str(recover_time))
        pickle.dump([tucker_arms, tucker_core, recover_time],
                    open("data/walk_2pass_tucker_k" + str(k) + "_r" + str(50_50_3) + ".pickle", 'wb'))
