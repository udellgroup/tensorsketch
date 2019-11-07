import numpy as np
def eval_rerr(X, X_hat, X0=None):
    """
    :param X: tensor, X0 or X0+noise
    :param X_hat: output for apporoximation
    :param X0: true signal, tensor
    :return: the relative error = ||X- X_hat||_F/ ||X_0||_F
    """
    if X0 is not None:
        error = X0 - X_hat
        return np.linalg.norm(error.reshape(np.size(error), 1), 'fro') / \
           np.linalg.norm(X0.reshape(np.size(X0), 1), 'fro')
    error = X - X_hat
    return np.linalg.norm(error.reshape(np.size(error), 1), 'fro') / \
           np.linalg.norm(X0.reshape(np.size(X), 1), 'fro')