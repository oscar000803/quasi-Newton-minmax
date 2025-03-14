import torch
torch.manual_seed(42)
from run import autograd
import os
import time
import csv
import math
from collections import deque

class LBFGS_param:
    def __init__(self):
        self.init = 0
        self.dx = None
        self.rho = deque()
        self.diff_x = deque()
        self.diff_dx = deque()

# Define save folder
def get_save_folder(n, m, opt_x, opt_y, lr_x, lr_y, num_y, mem_y = 0):
    if(mem_y):
        return "./SCSC/{}-{}/{}-{}-{}-{}-{}-{}".format(n, m, opt_x, opt_y, lr_x, lr_y, num_y, mem_y)
    return "./SCSC/{}-{}/{}-{}-{}-{}-{}".format(n, m, opt_x, opt_y, lr_x, lr_y, num_y)

def train(epoch, stoptime, stop, n, m, opt_x, opt_y, lr_x, lr_y, num_y, pretrain, mem_y = 0, device = "cuda:0"):
    # Make save_folder
    save_folder = get_save_folder(n, m, opt_x, opt_y, lr_x, lr_y, num_y, mem_y = False)

    # Initialize lists to save value
    time_seq = []
    px_norm = []
    py_norm = []

    # Initilize coefficients
    b = torch.zeros(m, device = device) # torch.randn(m, requires_grad=False)
    A = torch.randn(m, n, requires_grad=False, device = device)
    lmbda = 1/n  # Regularization parameter

    # Initialize variables
    x = 10/n*torch.randn(n, requires_grad=True, device = device)
    y = 10/m*torch.randn(m, requires_grad=True, device = device)
    print(torch.norm(x).item(), ", ", torch.norm(y).item())

    # Define objective function
    def f(x,y):
        return (-torch.norm(y) ** 2 / 2 - b @ y + y @ A @ x) / n + lmbda / 2 * torch.norm(x) ** 2

    # Define partial derivatives
    def px(x, y):
        z = f(x, y)
        return torch.autograd.grad(z, x, create_graph=True)[0]

    def py(x, y):
        z = f(x, y)
        return torch.autograd.grad(z, y, create_graph=True)[0]

    def pxx(x, y):
        h = []
        grad_x = px(x, y)
        for g in grad_x:
            h.append(torch.autograd.grad(g, x, retain_graph=True)[0])
        return torch.stack(h)

    def pyy(x, y):
        h = []
        grad_y = py(x, y)
        for g in grad_y:
            h.append(torch.autograd.grad(g, y, retain_graph=True)[0])
        return torch.stack(h)

    def pxy(x, y):
        h = []
        grad_x = px(x, y)
        for g in grad_x:
            h.append(torch.autograd.grad(g, y, retain_graph=True)[0])
        return torch.stack(h)
    
    def pyx(x, y):
        h = []
        grad_y = py(x, y)
        for g in grad_y:
            h.append(torch.autograd.grad(g, x, retain_graph=True)[0])
        return torch.stack(h)

    def pyy_inv(x, y):
        return torch.linalg.inv(pyy(x, y))

    def Dxx_inv(x, y):
        return torch.linalg.inv(pxx(x,y) - torch.matmul(torch.matmul(pxy(x,y), torch.linalg.inv(pyy(x,y))), pyx(x,y)))

    px_norm.append(torch.norm(px(x,y)).item())
    py_norm.append(torch.norm(py(x,y)).item())
    time_seq.append(0)
    start_time = time.time()
    pre_time = start_time

    for i in range(pretrain):
        x = x - 1.0 * px(x,y)
        y = y + 1.0 * py(x,y)
        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)

    # Optimization 

    ## prepare BFGS for outer problem
    H_o = torch.eye(n, device = device)
    LBFGS_o = LBFGS_param()
    
    num_ite = 1000000
    if stop == "epoch":
        num_ite = epoch
    # Optimization loop
    for i in range(num_ite):

        # outer problem

        if opt_x == "GD":
            x = x - lr_x * px(x,y)
        elif opt_x == "Newton":
            x = x - lr_x * torch.matmul(Dxx_inv(x,y), px(x,y).view(-1, 1)).squeeze()
        elif opt_x == "BFGS":
            _px = px(x,y)
            s = - lr_x * torch.matmul(H_o, _px.view(-1,1)).squeeze()
            x = x + s
            _px_new = px(x,y)
            g = _px_new - _px

            s = s.detach()
            g = g.detach()

            rho = 1 / torch.matmul(g, s.view(-1,1))
            V = (torch.eye(n, device = device, requires_grad=False) - rho * torch.matmul(s.view(-1,1), g.view(1,-1)))
            H_o = torch.matmul(torch.matmul(V, H_o), torch.transpose(V, 0, 1)) + rho * torch.matmul(s.view(-1,1), s.view(1,-1))
            H_o.detach()
        elif opt_x == "LBFGS":
            _px = px(x,y)
            q = _px
            if len(LBFGS_o.diff_x) == 0:
                x = x - lr_x * q
                s = - lr_x * q
                g = px(x,y) - _px
                s = s.detach()
                g = g.detach()
                LBFGS_o.diff_dx.append(g)
                LBFGS_o.diff_x.append(s)
            else:
                alphas = []
                for S, G in zip(reversed(LBFGS_o.diff_x), reversed(LBFGS_o.diff_dx)):
                    alpha = torch.matmul(S, q)/torch.matmul(G, S)
                    q = q - alpha * G
                    alphas.append(alpha)
                r = q
                for S, G, alpha in zip(LBFGS_o.diff_x, LBFGS_o.diff_dx, reversed(alphas)):
                    beta = torch.matmul(G, r)/torch.matmul(G, S)
                    r = r + (alpha - beta) * S
                s = - lr_x * r
                x = x + s
                g = px(x,y) - _px
                s = s.detach()
                g = g.detach()
                LBFGS_o.diff_dx.append(g)
                LBFGS_o.diff_x.append(s)
                if len(LBFGS_o.diff_dx) > mem_y:
                    LBFGS_o.diff_x.popleft()
                    LBFGS_o.diff_dx.popleft()


        # prepare BFGS/LBFGS for inner problem
        H_i = torch.eye(m, device = device)
        
        LBFGS_i = LBFGS_param()

        # inner problem

        for j in range(num_y):
            if opt_y == "GD":
                y = y + lr_y * py(x,y)
            elif opt_y == "Newton":
                y = y - lr_y * torch.matmul(pyy_inv(x,y), py(x,y).view(-1, 1)).squeeze()
            elif opt_y == "BFGS":
                _py = py(x,y)
                s = lr_y * torch.matmul(H_i, _py.view(-1,1)).squeeze()
                y = y + s
                _py_new = py(x,y)
                g = - _py_new + _py

                s = s.detach()
                g = g.detach()

                rho = 1 / torch.matmul(g, s.view(-1,1))
                V = (torch.eye(n, device = device, requires_grad=False) - rho * torch.matmul(s.view(-1,1), g.view(1,-1)))
                H_i = torch.matmul(torch.matmul(V, H_i), torch.transpose(V, 0, 1)) + rho * torch.matmul(s.view(-1,1), s.view(1,-1))
                H_i.detach()
            elif opt_y == "LBFGS":
                _py = py(x,y)
                q = - _py
                if len(LBFGS_i.diff_x) == 0:
                    y = y - lr_y * q
                    s = - lr_y * q
                    g = - py(x,y) + _py
                    s = s.detach()
                    g = g.detach()
                    LBFGS_i.diff_dx.append(g)
                    LBFGS_i.diff_x.append(s)
                else:
                    alphas = []
                    for S, G in zip(reversed(LBFGS_i.diff_x), reversed(LBFGS_i.diff_dx)):
                        alpha = torch.matmul(S, q)/torch.matmul(G, S)
                        q = q - alpha * G
                        alphas.append(alpha)
                    r = q
                    for S, G, alpha in zip(LBFGS_i.diff_x, LBFGS_i.diff_dx, reversed(alphas)):
                        beta = torch.matmul(G, r)/torch.matmul(G, S)
                        r = r + (alpha - beta) * S
                    s = -lr_y * r
                    y = y + s
                    g = - py(x,y) + _py
                    s = s.detach()
                    g = g.detach()
                    LBFGS_i.diff_dx.append(g)
                    LBFGS_i.diff_x.append(s)
                    if len(LBFGS_i.diff_dx) > mem_y:
                        LBFGS_i.diff_x.popleft()
                        LBFGS_i.diff_dx.popleft()

        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)

        # Log progress
        if math.isnan(torch.norm(px(x,y)).item()) or math.isnan(torch.norm(py(x,y)).item()) or torch.norm(px(x,y)).item() == 0 or torch.norm(py(x,y)).item() == 0:
            print("nan")
            break
        px_norm.append(torch.norm(px(x,y)).item())
        py_norm.append(torch.norm(py(x,y)).item())
        cur_time = time.time()
        time_seq.append(cur_time - start_time)
        if i % 100 == 0 or i == epoch - 1:
            print(f"Iteration {i}: dxf(x, y) = {torch.norm(px(x,y)).item()}, dyf(x, y) = {torch.norm(py(x,y)).item()}, time = {cur_time - start_time}, {cur_time - pre_time}")
            pre_time = cur_time
        if stop == "time" and cur_time - start_time > stoptime:
            break
        elif stop == "both" and cur_time - start_time > stoptime and i+1 >= epoch:
            break

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, "time.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        # Write each item as a new row
        for px, py, t in zip(px_norm, py_norm, time_seq):
            writer.writerow([px, py, t])



import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--time", type=int, default=0)
parser.add_argument("--stop", type=str, default="epoch")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--m", type=int, default=10)
parser.add_argument("--opt_x", type=str, default="GD")
parser.add_argument("--opt_y", type=str, default="GD")
parser.add_argument("--lr_x", type=float, default=0.05)
parser.add_argument("--lr_y", type=float, default=0.05)
parser.add_argument("--num_y", type=int, default=1)
parser.add_argument("--mem_y", type=int, default=0)
parser.add_argument("--pretrain", type=int, default=0)

args = parser.parse_args()

train(epoch = args.epoch, stoptime = args.time, stop = args.stop,
      n = args.n, m = args.m,
      opt_x = args.opt_x, opt_y = args.opt_y,
      lr_x = args.lr_x, lr_y = args.lr_y,
      num_y = args.num_y, mem_y = args.mem_y,
      pretrain = args.pretrain,
      device = "cuda:0")