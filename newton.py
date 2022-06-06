import torch
from torch.autograd.functional import jacobian


def get_gradient(func, input_tensor):
    x_g = input_tensor.clone().requires_grad_(True)
    h = func(x_g)
    h.backward()
    # print(x_g.grad)
    return x_g.grad.detach()


def get_hessian_batched(func, input_tensor):
    """
    Computes the batched Hessian (one hessian per row of `input_tensor`)
    :param func: function must operate row-wise on input_tensor
    :param input_tensor: arbitrary shape with first dimension as batch. Tested only for 2D tensors so far
    :return: Hessian tensor of size (n, input_tensor.size)
    """
    x_h = input_tensor.clone().requires_grad_(True)

    def get_sum_of_gradients(x):
        h = f(x)
        return torch.autograd.grad(h, x, create_graph=True)[0].sum(0)

    hessian = jacobian(get_sum_of_gradients, x, vectorize=True).swapaxes(0, 1)
    sym_hessian = (hessian + hessian.swapaxes(-1, -2)) / 2
    return sym_hessian


def batched_newton_method(
        func, init, max_iter=100, lr=0.9,
        xtol=1e-16, gtol=1e-12,
        ftol_up=1e0, ftol_down=1e-16,
        verbose=True
):
    """
    Returns 
    argmin{Z} func(Z)
    using Newton's method
    
    Takes as input `func` which is a function (or any object with a __call__ method)
    and `init` which is initialization
    """
    
    terminated = False
    opt_variable = init.clone()
    # try:
    f_init = func(init)
    f_prev = torch.tensor([1]) / 0
    update = torch.zeros_like(init)
    for t in range(max_iter):
        fval = func(opt_variable)
        if fval > f_prev + ftol_up:
            if verbose: print(f'Terminated after {t} iterations since '
                              f'function value increased '
                              f'from {f_prev:.2f} to {fval:.2f}')
            terminated = True
            break
        elif fval > f_prev - ftol_down:
            if verbose: print(f'Terminated after {t} iterations since '
                              f'function value has not decreased by {ftol_down} '
                              f'in last iteration')
            terminated = True
            break
        gradient = get_gradient(func, opt_variable)
        hessian = get_hessian_batched(func, opt_variable)
        if torch.norm(gradient) / gradient.shape[0] < gtol:
            if verbose: print(f'Terminated after {t} iterations since '
                              f'gradient tolerance reached')
            terminated = True
            break
        update = -lr * torch.linalg.solve(hessian, gradient)
        if torch.norm(update) ** 2 / update.shape[0] < xtol:
            if verbose: print(f'Terminated after {t} iterations since no progress made')
            terminated = True
            break
        f_prev = fval
        opt_variable = opt_variable + update
    if not terminated:
        if verbose: print(f'Completed {t + 1} iterations of Newton`s method\n'
                          f'Function value reduced from {f_init:.2f} to {func(opt_variable):.2f}')
    return opt_variable, f_prev, gradient, hessian, update
