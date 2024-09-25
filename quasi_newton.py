import torch
import numpy as np

def flat(params):
    return torch.cat([p.flatten() for p in params])

class BFGS_param:
    def __init__(self):
        self.approximate_Hessian = None
        self.diff_x = None
        self.dx = None

def BFGS_update(BFGS, step_size, model, loss_fn, autograd):
    grad = autograd(loss_fn(), model.parameters())
    param_flatten = flat(model.parameters())
    # print("pf",param_flatten)
    grad_flatten = flat(grad)

    if BFGS.approximate_Hessian == None:
        total_p = sum(p.numel() for p in model.parameters())
        BFGS.approximate_Hessian = torch.eye(total_p, device=next(model.parameters()).device).double()
        BFGS.dx = grad_flatten
    else:
        diff_dx = grad_flatten - BFGS.dx
        rho = torch.matmul(diff_dx, BFGS.diff_x)
        V = torch.matmul(diff_dx.view(-1, 1), BFGS.diff_x.view(1, -1))
        V = torch.eye(V.shape[0], device=next(model.parameters()).device) - V
        BFGS.approximate_Hessian = torch.matmul(torch.matmul(V.t(), BFGS.approximate_Hessian), V) + rho * torch.matmul(BFGS.diff_x.view(-1, 1), BFGS.diff_x.view(1, -1))
        BFGS.dx = grad_flatten

    p = torch.matmul(BFGS.approximate_Hessian, grad_flatten)
    s = step_size * p
    s_chunks = torch.split(s, [p.numel() for p in model.parameters()])
    reshape_s = [chunk.view(shape) for chunk, shape in zip(s_chunks, [p.shape for p in model.parameters()])]
    BFGS.diff_x = s
    # print(reshape_s)
    return reshape_s