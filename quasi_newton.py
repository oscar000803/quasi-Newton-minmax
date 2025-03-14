import torch
import numpy as np
import math
from collections import deque

def compute_hessian(loss, model):
    params = list(model.parameters())
    hessian = []
    
    # First-order gradient
    grad_params = torch.autograd.grad(loss, params, create_graph=True)
    
    # Loop over each gradient
    for grad in grad_params:
        hessian_row = []
        
        # Loop over each element of the gradient
        for g in grad.view(-1):
            # Compute second derivatives
            second_grad = torch.autograd.grad(g, params, retain_graph=True)
            hessian_row.append(torch.cat([sg.view(-1) for sg in second_grad]).unsqueeze(0))
        
        # Stack to form a Hessian row
        hessian.append(torch.cat(hessian_row, dim=0))
    
    # Concatenate Hessian rows
    return torch.cat(hessian, dim=0)

def flat(params):
    return torch.cat([p.flatten() for p in params])

class BFGS_param:
    def __init__(self):
        self.approximate_Hessian = None
        self.diff_x = None
        self.dx = None

def BFGS_update(BFGS, step_size, model, loss_fn, autograd):
    grad = autograd(loss_fn(), model.parameters())
    grad_flatten = flat(grad)
    if torch.equal(grad_flatten, torch.zeros_like(grad_flatten)):
        return 

    if BFGS.approximate_Hessian == None:
        # initial I
        total_p = sum(p.numel() for p in model.parameters())
        BFGS.approximate_Hessian = torch.eye(total_p, device=next(model.parameters()).device).double()
        # initial H
        # BFGS.approximate_Hessian = compute_hessian(loss_fn(), model)
    else:
        if torch.equal(BFGS.dx, grad_flatten):
            return 
        diff_dx = grad_flatten - BFGS.dx
        if torch.equal(grad_flatten, BFGS.dx):
            return 
        rho = 1/torch.matmul(diff_dx, BFGS.diff_x)
        V = rho * torch.matmul(diff_dx.view(-1, 1), BFGS.diff_x.view(1, -1))
        V = torch.eye(V.shape[0], device=next(model.parameters()).device) - V
        BFGS.approximate_Hessian = torch.matmul(torch.matmul(V.t(), BFGS.approximate_Hessian), V) + rho * torch.matmul(BFGS.diff_x.view(-1, 1), BFGS.diff_x.view(1, -1))
        
    BFGS.dx = grad_flatten
    p = - torch.matmul(BFGS.approximate_Hessian, grad_flatten)
    s = step_size * p
    s_chunks = torch.split(s, [p.numel() for p in model.parameters()])
    reshape_s = [chunk.view(shape) for chunk, shape in zip(s_chunks, [p.shape for p in model.parameters()])]
    BFGS.diff_x = s
    if(math.isnan(reshape_s[0][0])):
        return
    return reshape_s

class LBFGS_param:
    def __init__(self):
        self.init = 0
        self.dx = None
        self.rho = deque()
        self.diff_x = deque()
        self.diff_dx = deque()
        
def LBFGS_update(LBFGS, step_size, model, loss_fn, autograd, save):
    grad = autograd(loss_fn(), model.parameters())
    grad_flatten = flat(grad)
    if torch.equal(grad_flatten, torch.zeros_like(grad_flatten)):
        return 

    p = 0
    if LBFGS.init == 0:
        LBFGS.init = 1
        p = - grad_flatten
    else:
        if torch.equal(LBFGS.dx, grad_flatten):
            return 
        if len(LBFGS.diff_dx) == save:
            LBFGS.diff_dx.popleft()
            LBFGS.rho.popleft()
        LBFGS.diff_dx.append(grad_flatten - LBFGS.dx)
        LBFGS.rho.append(1/torch.matmul(LBFGS.diff_dx[-1], LBFGS.diff_x[-1]))
        H = torch.matmul(LBFGS.diff_x[-1], LBFGS.diff_dx[-1]) / torch.matmul(LBFGS.diff_dx[-1], LBFGS.diff_dx[-1])

        q = grad_flatten
        alphas = []
        for rho, s, y in zip(reversed(LBFGS.rho), reversed(LBFGS.diff_x), reversed(LBFGS.diff_dx)):
            alpha = rho*torch.matmul(s, q)
            alphas.append(alpha)
            q = q - alpha*y
        # r = H*q
        r = q
        for rho, s, y, alpha in zip(LBFGS.rho, LBFGS.diff_x, LBFGS.diff_dx, reversed(alphas)):
            beta = rho*torch.matmul(y, r)
            r = r + (alpha - beta)*s
        p = - r

    LBFGS.dx = grad_flatten
    s = step_size * p
    s_chunks = torch.split(s, [p.numel() for p in model.parameters()])
    reshape_s = [chunk.view(shape) for chunk, shape in zip(s_chunks, [p.shape for p in model.parameters()])]

    if len(LBFGS.diff_x) == save:
        LBFGS.diff_x.popleft()
    LBFGS.diff_x.append(s)    

    return reshape_s