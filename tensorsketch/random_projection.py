import numpy as np
import tensorly as tl


def random_matrix_generator(n, k, typ="g", target='col'):
    """
    routine for usage: A \Omega or \Omega^\top x : n >> m
    :param n: first dimension of random matrix to be generated
    :param k: second dimension of random matrix to be generated
    :param type:
    :param target:  for column preservation or length preservation,
    for column preservation, we do not need standardization
    :return:
    """

    if isinstance(n, list):
        n = np.prod(n)

    types = set(['g', 'u', 'sp0', 'sp1'])
    assert typ in types, "please aset your type of random variable correctly"
    assert target in ['col', 'vec'], "target can only be col or vec"
    if typ == 'g':
        Omega = np.random.normal(0, 1, size=(n, k))
    elif typ == 'u':
        Omega = np.random.uniform(low=-1, high=1, size=(n, k)) * np.sqrt(3)
    elif typ == 'sp0':
        Omega = np.random.choice([-1, 0, 1], size=(n, k), p=[1 / 6, 2 / 3, 1 / 6]) * np.sqrt(3)
    elif typ == 'sp1':
        Omega = np.random.choice([-1, 0, 1], size=(n, k), p= \
            [1 / (2 * np.sqrt(n)), 1 - 1 / np.sqrt(n), 1 / (2 * np.sqrt(n))]) * np.sqrt(np.sqrt(n))
    if target == 'col':
        return Omega
    return Omega.transpose()/np.sqrt(k)


def tensor_random_matrix_generator(n_arr, k, typ="g", target='col'):
    """
    routine for usage: A \Omega or \Omega^\top x : n >> m
    :param n_arr: first dimension of random matrix to be generated as a list
    n1*...n_{I+1}*...*n_N
    :param k: second dimension of random matrix to be generated
    :param type:
    :param target:  for column preservation or length preservation,
    for column preservation, we do not need standardization
    :return:
    """
    if not isinstance(n_arr, list):
        raise Exception("type of first parameter must be list")

    types = set(['g', 'u', 'sp0', 'sp1'])
    assert typ in types, "please aset your type of random variable correctly"
    assert target in ['col', 'vec'], "target can only be col or vec"
    Omegas = []
    for n in n_arr:
        if typ == 'g':
            Omega = np.random.normal(0, 1, size=(n, k))
        elif typ == 'u':
            Omega = np.random.uniform(low=-1, high=1, size=(n, k)) * np.sqrt(3)
        elif typ == 'sp0':
            Omega = np.random.choice([-1, 0, 1], size=(n, k), p=[1 / 6, 2 / 3, 1 / 6]) * np.sqrt(3)
        elif typ == 'sp1':
            Omega = np.random.choice([-1, 0, 1], size=(n, k), p= \
                [1 / (2 * np.sqrt(n)), 1 - 1 / np.sqrt(n), 1 / (2 * np.sqrt(n))]) * np.sqrt(np.sqrt(n))
        Omegas.append(Omega)
    if target == 'col':
        return tl.tenalg.khatri_rao(Omegas)
    return tl.tenalg.khatri_rao(Omegas).transpose()/np.sqrt(k)





if __name__ == "__main__":


    def test_khatri_rao():
        A = np.asarray([[1,2],[2,3],[1,2]])
        print(A)
        print(tl.tenalg.khatri_rao([A, A]))

    # test_khatri_rao()

    def test_dimension():
        omega = random_matrix_generator(100, 3, typ="g", target='col')
        print(omega.shape)
        omega = random_matrix_generator(100, 3, typ="g", target='vec')
        print(omega.shape)
        omega = tensor_random_matrix_generator([20, 20], 3, typ="g", target='col')
        print(omega.shape)
        omega = tensor_random_matrix_generator([20, 20], 3, typ="g", target='vec')
        print(omega.shape)

    test_dimension()

    def test_normalization():

        omega1 = random_matrix_generator(100, 3, typ="g", target='col')
        omega2 = random_matrix_generator(100, 3, typ="g", target='col')
        print("difference for two generated random matrix is {}".format(np.linalg.norm(omega1-omega2)))

        # test vector
        print("test normal random projection")
        x = np.random.uniform(0, 1, 100)
        original_norm = np.linalg.norm(x)
        res = []
        for _ in range(100):
            omega = random_matrix_generator(100, 20, typ="g", target='vec')
            res.append(np.linalg.norm(omega@x))
        ave_norm = np.mean(res)
        print(f"original norm:{original_norm}, {ave_norm}")

        res = []
        print("test tensor random projection")
        for _ in range(100):
            omega = tensor_random_matrix_generator([10, 10], 20, typ="g", target='vec')
            res.append(np.linalg.norm(omega@x))
        ave_norm = np.mean(res)
        print(f"original norm:{original_norm}, {ave_norm}")


    test_normalization()