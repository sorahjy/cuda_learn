# -*- coding: UTF-8 -*-
import os
os.environ['NVCC_APPEND_FLAGS'] = "-allow-unsupported-compiler"
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="add2",
                   extra_include_paths=["include"],
                   sources=["add2.cpp", "hjy_kernel/add2.cu"],
                   verbose=True)
#  计算 torch.add
n = 3000
m = 4000

device = "cuda:0"

a = torch.rand(n, 1, device=device)
b = torch.ones(1, device=device)
c = torch.rand(m, device=device)
cuda_d = torch.zeros(n, m, device=device)

ntest = 50


def show_time(func):
    st = 0
    res = 0
    # GPU warm up
    # for _ in range(1):
    func()

    st = time.time()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        # torch.cuda.synchronize(device="cuda:3")
        # start_time = time.time()
        func()
        # torch.cuda.synchronize(device="cuda:3")
        # end_time = time.time()

    res = (time.time() - st) / ntest * 1e6
    return res


def run_cuda():
    cuda_module.torch_launch_add2(cuda_d, a, b, c, n, m)
    return cuda_d


def run_torch():
    return torch.add(c, a, alpha=1)


res_t = run_torch()
res_c = run_cuda()

print(res_t)
print(res_c)

flag = torch.allclose(res_c, res_t)
print(flag)

print("Running cuda...")
cuda_time = show_time(run_cuda)
# print('ans:\n',_)
print("Cuda time:  {:.3f}us".format(cuda_time))

print("Running torch...")
torch_time = show_time(run_torch)
# print('ans:\n',_)
print("Torch time:  {:.3f}us".format(torch_time))

# yum install -y nsight-systems
# nsys profile python3 test.py
