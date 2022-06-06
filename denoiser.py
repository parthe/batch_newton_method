import torch
from rescaled_square_loss.newton import batched_newton_method
from torch.nn.functional import one_hot, relu
mse_entropy_loss = torch.nn.MSELoss()

# ||Z||_M^2
# then Zhat = RG @ (G + M).inverse()
# and hessian is (G + M) for each row.

def gen_frobenius_norm_squared(X, G):
    return torch.trace(X.T @ X @ G)

def relu_denoiser(PG, G, R, y, verbose=False):
    """
    Returns 
    argmin{Z} 
        mse_entropy_loss(Z, y)
        0.5 * ||Z||_Gamma^2 
        - <Z,RG> 
    
    Takes as input RG which is (R @ G)
    """
    func = lambda Z : (
            mse_entropy_loss(relu(Z), one_hot(y).float()*R)
            + 0.5 * gen_frobenius_norm_squared(Z, G)
            + torch.trace(Z.T @ PG)
    )
    Zhat, _, _, hessian, _ = batched_newton_method(func, PG, verbose=verbose)
    # Zhat is of size (n, K)
    # hessian is of size (n, K, K)
    
    eta = hessian.inverse().mean(0).inverse()
    return Zhat, (eta + eta.T)/2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    n, k = 1000, 20
    y = torch.randint(0, k, (n,))
    P = torch.randn(n, k)
    G = torch.eye(k) * .1
    R = 1000

    zhat, H = relu_denoiser(P @ G, G, R, y)
    # plt.scatter(P.norm(dim=1), zhat.norm(dim=1))
    plt.scatter(P[one_hot(y) == 1], zhat[one_hot(y) == 1], color='g')
    plt.scatter(P[one_hot(y) == 0], zhat[one_hot(y) == 0], color='r')
    plt.show()