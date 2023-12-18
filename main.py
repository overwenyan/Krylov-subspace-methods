#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Reinaldo Astudillo and Martin B. van Gijzen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# from __future__ import division, print_function, absolute_import

from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import gmres
from scipy.io import mmread

from idrs import idrs
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time


if __name__ == '__main__':
    # This Python script defines a 3D discretised
    # convection-diffusion-reaction problem on the unit cube.
    # The problem is solved with IDR(1), IDR(2), IDR(4), IDR(8).
    # This script is based of the following MATLAB script:
    # http://ta.twi.tudelft.nl/nw/users/gijzen/example_idrs.m

    
    file_name = 'data/can__838.mtx'
    A = mmread(file_name)

    n = A.shape[0]
    # print(n)
    b = np.ones(n) / pow(n,2)

    msg = "Method {:8} Time = {:6.3f} Matvec = {:d} Residual = {:g}"
    tol = 1e-8
    maxit = 5000
    x0 = np.zeros([n, 1])
    bnrm2 = np.linalg.norm(b)

    

    # def callback_(x):
    #     global matvec
    #     matvec = matvec + 1
    
    def residual(x):
        return np.linalg.norm(b - A.dot(x))/bnrm2
    
    matvec = 0
    residuals = []
    def callback_(x):
        global residuals, matvec
        matvec += 1
        
        res = residual(x)
        residuals.append(res)

    matvec = 0
    residuals = []
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=1, callback=callback_)
    # print(sols)
    elapsed_time = time.time() - t
    print(msg.format('IDR(1)', elapsed_time, matvec, residual(x)))
    plt.scatter(list(range(matvec)), residuals)
    # plt.show()

    matvec = 0
    residuals = []
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=2, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('IDR(2)', elapsed_time, matvec, residual(x)))

    matvec = 0
    residuals = []
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=4, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('IDR(4)', elapsed_time, matvec, residual(x)))

    matvec = 0
    residuals = []
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=8, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('IDR(8)', elapsed_time, matvec, residual(x)))

    matvec = 0
    residuals = []
    t = time.time()
    xb1, info = bicg(A, b, tol=1e-8, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('bicg', elapsed_time, 2 * matvec, residual(xb1)))
    
    matvec = 0
    residuals = []
    t = time.time()
    xb2, info = bicgstab(A, b, tol=1e-8, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('bicgstab', elapsed_time, 2 * matvec, residual(xb2)))

    matvec = 0
    residuals = []
    t = time.time()
    xg, info = gmres(A, b, tol=1e-8, restart=200, callback=callback_)
    elapsed_time = time.time() - t
    print(msg.format('gmres', elapsed_time, matvec, residual(xg)))
