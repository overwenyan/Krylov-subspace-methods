#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import gmres, cg
from scipy.sparse.linalg import norm, inv
from scipy.io import mmread

from cg_solver import cg_solver

from idrs import idrs
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

from utils import *

import argparse

# def run_krylov(A, b, method='idr'):
def get_method_name(args):
    method_name = ''
    if args.method == 'cg':
        method_name = args.method.upper()
    elif args.method == 'bicg':
        method_name = 'BiCG'
    elif args.method == 'bicgstab':
        method_name = 'BiCGSTAB'
    elif args.method == 'gmres':
        # if args.restarted:
        method_name = 'GMRES({})'.format(args.restart)
        # else:
            # method_name = 'GMRES'
    elif args.method == 'idr':
        method_name = 'IDR({})'.format(args.num_s)
    else:
        pass
    
    return method_name

def plot_residuals(file_name, residual_dict):
    matrix_name, sep, tail = file_name.partition('.')
    matrix_name = matrix_name.replace('_', '')
    mat_title = matrix_name.upper()
    
    plt.figure()
    plt.title(mat_title)
    for method in residual_dict:
        # print(method)
        reses = residual_dict[method]
        if method in ['GMRES(20)', 'GMRES(50)']:
            print(reses[0])
            reses = reses / reses[0]
        length = len(reses)
        # plt.plot(list(range(length)), reses, label=method)
        
        plt.loglog(list(range(length)), reses, label=method)
    
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Residual')
    
    plt.savefig(f'figs/res_{mat_title}.jpg')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_name", type=str, default='gmres', help="matrix name.")
    parser.add_argument("--method", type=str, default='gmres', choices=['cg', 'gmres', 'bicg', 'bicgstab', 'idr'], help="krylov subspace methods.")
    parser.add_argument("--tol", type=float, default=1e-8, help="tolerance for convergence test.")
    parser.add_argument("--maxit", type=int, default=5000, help="max iteration number.")
    # parser.add_argument("--restarted", type=bool, default=False, help="restarted gmres")
    parser.add_argument("--restart", type=int, default=200, help="restarted gmres iteration counts")
    parser.add_argument("--num_s", type=int, default=4, help="s for idr(s), dimension of shadow space.")
    args = parser.parse_args()
    
    folder_path = 'data/'
    # filenames = list_files_in_directory(folder_path)
    # matrix_file_name = filenames[7]
    matrix_file_name = args.matrix_name
    matrix_name, sep, tail = matrix_file_name.partition('.')
    matrix_name = matrix_name.replace('_', '')
    mat_title = matrix_name.upper()
    file_name = folder_path + matrix_file_name
    
    my_matrix = mmread(file_name)
    A = my_matrix
    my_array = my_matrix.A
    cond_number = np.linalg.cond(my_array)
    n = A.shape[0]
    b = np.ones(n) / pow(n,2)
    print("Matrix name:  {}, dimension: {}, $\kappa(A)={:.5g}$".format(mat_title, n, cond_number))
    
    
    msg = "Method {:8} Time = {:6.3f} Matvec = {:d} Residual = {:g}"
    msg = "{:8} & {:6.3f} & {:d} & {:g} \\\\"
    x0 = np.zeros([n, 1])
    # x0 = np.random.randint(low=0, high=1, size=(n,1))
    bnrm2 = np.linalg.norm(b)
    res0 = np.linalg.norm(b - A.dot(x0))
    residual_dict = {}
    
    def residual(x):
        return np.linalg.norm(b - A.dot(x))/bnrm2
    
    matvec = 0
    residuals = []
    def callback_(x):
        global matvec, residuals
        matvec = matvec + 1
        # res = residual(x)
        res = np.linalg.norm(b - A.dot(x)) / bnrm2 
        # res = np.linalg.norm(b - A.dot(x))/bnrm2
        residuals.append(res)

    # CG
    # matvec = 0
    # residuals = []
    # args.method = 'cg'
    # args.restart = 20
    # t = time.time()
    # xb2, info = cg(A, b, x0=x0, tol=args.tol, maxiter=args.maxit, callback=callback_)
    # elapsed_time = time.time() - t
    # method_name = get_method_name(args)
    # print(msg.format(method_name, elapsed_time, matvec, residual(xb2)))
    # residual_dict[method_name] = residuals
    
    # GMRES
    matvec = 0
    residuals = []
    args.method = 'gmres'
    args.restart = 20
    t = time.time()
    xb2, info = gmres(A, b, x0=x0, tol=args.tol, restart=args.restart, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    method_name = get_method_name(args)
    print(msg.format(method_name, elapsed_time, matvec, residual(xb2)))
    # print(residuals[-1])
    # residuals = [res / np.log(res)  for res in residuals]
    residual_dict[method_name] = residuals 
    
    # GMRES
    matvec = 0
    residuals = []
    args.method = 'gmres'
    args.restart = 50
    t = time.time()
    x, info = gmres(A, b, x0=x0, tol=args.tol, restart=args.restart, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    method_name = get_method_name(args)
    print(msg.format(method_name, elapsed_time, matvec, residual(x)))
    # residual_dict[method_name] = residuals / residuals[-1] * residual(x)
    # residuals = [res / np.log(res)  for res in residuals]
    # for i in range(len(residuals)):
    #     residuals[i] = residuals[i] / residuals[0]
    # print(residuals[0])
    residual_dict[method_name] = residuals
    
    
    # BiCG
    matvec = 0
    residuals = []
    args.method = 'bicg'
    t = time.time()
    xb1, info = bicg(A, b, x0=x0, tol=args.tol, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    method_name = get_method_name(args)
    print(msg.format(method_name, elapsed_time, matvec, residual(xb1)))
    residual_dict[method_name] = residuals
    # plt.plot(list(range(matvec)), residuals)
    
    # BiCGSTAB
    matvec = 0
    residuals = []
    args.method = 'bicgstab'
    t = time.time()
    xb2, info = bicgstab(A, b, x0=x0, tol=args.tol, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    method_name = get_method_name(args)
    print(msg.format(method_name, elapsed_time, matvec, residual(xb2)))
    residual_dict[method_name] = residuals
    
    
    # IDR
    matvec = 0
    residuals = []
    args.method = 'idr'
    args.num_s = 4
    method_name = get_method_name(args)
    t = time.time()
    x, info = idrs(A, b, x0=x0, tol=args.tol, s=args.num_s, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format(method_name, elapsed_time, matvec, residual(x)))
    residual_dict[method_name] = residuals
    
    # IDR
    matvec = 0
    residuals = []
    args.method = 'idr'
    args.num_s = 8
    method_name = get_method_name(args)
    t = time.time()
    x, info = idrs(A, b, x0=x0, tol=args.tol, s=args.num_s, maxiter=args.maxit, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format(method_name, elapsed_time, matvec, residual(x)))
    residual_dict[method_name] = residuals
    
    
    plot_residuals(matrix_file_name, residual_dict)
    
