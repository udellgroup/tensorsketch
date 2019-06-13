import numpy as np
from scipy import fftpack
import tensorly as tl

tl.set_backend('numpy')

class TensorInfoBucket(object):
    def __init__(self, tensor_shape, ks, ranks, ss = []):
        '''
        Information of the original tensor X
        :k,s: integer
        :ranks: n-darray for the ranks of X
        '''
        self.tensor_shape = tensor_shape
        self.ks = ks
        self.ranks = ranks
        self.ss = ss

    def get_info(self):
        return self.tensor_shape, self.ks, self.ranks, self.ss

class RandomInfoBucket(object):
    '''
    Information for generating randomized linear maps
    '''
    def __init__(self, std=1, typ='g', random_seed = 0, sparse_factor = 0.1):
        self.std = std
        self.typ = typ
        self.random_seed = random_seed
        self.sparse_factor = sparse_factor

    def get_info(self):
        return self.std, self.typ, self.random_seed, self.sparse_factor

def random_matrix_generator(m, n, Rinfo_bucket):
    '''
    Generate random matrix of size m x n
    :param m: length
    :param n: width
    :param Rinfo_bucket: parameters fo generating the random matrix: std (standard devidation for each entry); typ
    (u: uniform; g: Gaussian; sp: sparsity=sparse_factor; sp0: sparsity=2/3;sp1:sparsity=1-1/sqrt(n) )
    :return: random matrix
    '''
    std, typ, random_seed, sparse_factor = Rinfo_bucket.get_info()
    np.random.seed(random_seed)
    types = set(['g', 'u', 'sp', 'sp0', 'sp1'])
    assert typ in types, "please aset your type of random variable correctly"

    if typ == 'g':
        return np.random.normal(0,1, size = (m,n))*std
    elif typ == 'u':
        return np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std
    elif typ == 'sp':
        return np.random.choice([-1,0,1], size = (m,n), p = [sparse_factor/2, \
            1- sparse_factor, sparse_factor/2])*np.sqrt(1/sparse_factor)*std
    elif typ == 'sp0': 
        return np.random.choice([-1,0,1], size = (m,n), p = [1/6, 2/3,1/6])*np.sqrt(3)*std 
    elif typ == 'sp1':
        return np.random.choice([-1, 0, 1], size = (m,n), p = \
            [1/(2*np.sqrt(n)), 1- 1/np.sqrt(n), 1/(2*np.sqrt(n))])*np.sqrt(np.sqrt(n))*std

def ssrft(k,X, seed = 1, mult = "right"):
    '''
    Perform a SSRFT transform
    :param k: reduced dimension
    :param X: matrix
    :param seed: random seed
    :param mult: left reduce or right reduce the matrix to dimension k
    :return: Reduced matrix after ssrft transform
    '''
    np.random.seed(seed)
    m, n = X.shape 
    if mult == "left": 
        perm1 = np.random.permutation(m) 
        perm2 = np.random.permutation(m)  
        coord = np.random.permutation(m)[0:k] 
        sign1 = np.random.choice([-1, 1], size = m)
        sign2 = np.random.choice([-1, 1], size = m) 
        result = fftpack.dct(sign1.reshape(m,1)*X[perm1,:])
        result = fftpack.dct(sign2.reshape(m,1)*result[perm2,:])
        return result[coord,:]
    if mult == "right": 
        perm1 = np.random.permutation(n) 
        perm2 = np.random.permutation(n)  
        coord = np.random.permutation(n)[0:k] 
        sign1 = np.random.choice([-1, 1], size = n)
        sign2 = np.random.choice([-1, 1], size = n) 
        result = fftpack.dct(X[:, perm1]*sign1.reshape(1,n), axis = 1) 
        result = fftpack.dct(result[:, perm2]*sign2.reshape(1,n), axis = 1)
        return result[:,coord]



def ssrft_modeprod(k, X, mode, mult, seed = 1, fold = True): 
    '''
    SSRFT transform of a tensor
    :param k: reduced dimension
    :param X: data tensor
    :param mode: mode on which reduction happens
    :param mult: left reduce or right reduce
    :param seed: rand seed
    :param fold: whether fold the reduced the matrix back to a tensor
    '''
    shape = X.shape
    X = tl.unfold(X, mode=mode)
    X = ssrft(k, X, seed = seed, mult = mult)  
    shape = np.asarray(shape)
    shape[mode] = k
    if fold: 
        return tl.fold(X, mode = mode, shape = shape)
    else: 
        return X
    

def gprod(k, X, mode, seed = 1):
    '''
    Dimension reduction based on Tensor random projection (https://r2learning.github.io/assets/papers/CameraReadySubmission%2041.pdf)
    where each submatrix is standard Gaussian
    :param k: reduced dimension
    :param X: data tensor
    :param mode: mode on which dimension reduction happens
    :return: unfolding of the reduced tensor
    '''
    # Create a sketch of size k x I_(-mode) (Left mutliplication) and I_(mode) x k (Right multiplication) 
    np.random.seed(seed)
    I = X.shape
    randmat = []
    for idx, i in enumerate(I): 
        if idx == mode: 
            randmat.append(np.ones((1,k)))
        else: 
            #randmat.append((np.random.rand(i,k)-1)*2)
            randmat.append(np.random.normal(0,1,size = (i,k)))
    randmatprod = tl.tenalg.khatri_rao(randmat)
    return tl.unfold(X, mode = mode) @ randmatprod
def sp0prod(k, X, mode, seed = 1):
    # Create a sketch of size k x I_(-mode) (Left mutliplication) and I_(mode) x k (Right multiplication)
    '''
    Dimension reduction based on Tensor random projection (https://r2learning.github.io/assets/papers/CameraReadySubmission%2041.pdf)
    where each submatrix is sparse matrix with sparsity = 2/3
    :param k: reduced dimension
    :param X: data tensor
    :param mode: mode on which dimension reduction happens
    :return: unfolding of the reduced tensor
    '''
    np.random.seed(seed)
    I = X.shape
    randmat = []
    for idx, i in enumerate(I): 
        if idx == mode: 
            randmat.append(np.ones((1,k)))
        else: 
            randmat.append(np.random.choice([-1,0,1], size = (i,k), p = [1/6, 2/3,1/6])*np.sqrt(3))
    randmatprod = tl.tenalg.khatri_rao(randmat)
    return tl.unfold(X, mode = mode) @ randmatprod

def tensor_gen_help(core,arms):
    '''
    :param core: the core tensor in higher order svd s*s*...*s
    :param arms: those arms n*s
    :return:
    '''
    for i in np.arange(len(arms)):
        prod = tl.tenalg.mode_dot(core,arms[i],mode =i)
    return prod 


def generate_super_diagonal_tensor(diagonal_elems, dim):
    '''
    Generate super diagonal tensor of dimension = dim
    '''
    n = len(diagonal_elems)
    tensor = np.zeros(np.repeat(n, dim))
    for i in range(n):
        index = tuple([i for _ in range(dim)])
        tensor[index] = diagonal_elems[i]
    return tl.tensor(tensor)



def square_tensor_gen(n, r, dim = 3,  typ = 'id', noise_level = 0, seed = None, sparse_factor = 0.2):
    '''
    :param n: size of the tensor generated n*n*...*n
    :param r: rank of the tensor or equivalently, the size of core tensor
    :param dim: # of dimensions of the tensor, default set as 3
    :param typ: identity as core tensor or low rank as core tensor
    :param noise_level: sqrt(E||X||^2_F/E||error||^_F)
    :return: The tensor with noise, and The tensor without noise
    '''
    if seed: 
        np.random.seed(seed) 

    types = set(['id', 'lk', 'fpd', 'spd', 'sed', 'fed','slk'])
    assert typ in types, "please set your type of tensor correctly"
    total_num = np.power(n, dim)

    if typ == 'id':
        # identity
        elems = [1 for _ in range(r)]
        elems.extend([0 for _ in range(n-r)])
        noise = np.random.normal(0, 1, [n for _ in range(dim)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0 +noise*np.sqrt((noise_level**2)*r/total_num), X0
        
    if typ == 'spd':
        # Slow polynomial decay
        elems = [1 for _ in range(r)]
        elems.extend([1.0/i for i in range(2, n-r+2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == 'fpd':
        # Fast polynomial decay
        elems = [1 for _ in range(r)]
        elems.extend([1.0/(i*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'sed':
        # Slow exponential decay
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, -0.25*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'fed':
        # Fast Exponential decay
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, (-1.0)*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == "lk":
        # Low rank
        core_tensor = np.random.uniform(0,1,[r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in np.arange(dim):
            arm = np.random.normal(0,1,size = (n,r))
            arm, _ = np.linalg.qr(arm)
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
        true_signal_mag = np.linalg.norm(tensor)**2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        X = tensor + noise*np.sqrt((noise_level**2)*true_signal_mag/total_num)
        return X, tensor

    if typ == "slk":
        # Sparse low rank
        core_tensor = np.random.normal(0,1,[r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in np.arange(dim): 
            arm =  np.random.normal(0,1,size = (n,r))
            arm = arm*np.random.binomial(1,sparse_factor,size=(n,r))
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i) 
        true_signal_mag = np.linalg.norm(tensor)**2
        tensor0 = tensor
        tensor = tensor + np.random.normal(0,1,size = [n for _ in range(dim)])\
        *np.sqrt((noise_level**2)*true_signal_mag/total_num) 
        return tensor, tensor0 

def eval_rerr(X,X_hat,X0):
    # evaluate the relative error = ||X- X_hat||_F/ ||X_0||_F
    error = X-X_hat
    return np.linalg.norm(error.reshape(np.size(error),1),'fro')/ \
    np.linalg.norm(X0.reshape(np.size(X0),1),'fro')

if __name__ == "__main__":
    tl.set_backend('numpy')
    X = np.arange(25).reshape((5,5))
    Gmatrix = random_matrix_generator(3,5,RandomInfoBucket()) 
    print(np.dot(Gmatrix,X))
    print(ssrft(3,X, mult = "left").shape)
    X, _ = square_tensor_gen(5, 3, dim=3, typ='id', noise_level=0.1) 
    print(sum(X - tl.fold(tl.unfold(X,2),2,X.shape)))


    Y = np.arange(120).reshape((3,4,10)) 
    print(gprod(3,Y,1))





