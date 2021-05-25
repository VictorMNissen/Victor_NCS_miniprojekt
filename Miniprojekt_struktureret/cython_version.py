# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:32:55 2021

@author: Victor
"""
import matplotlib.pyplot as mplpp
import numpy as np
from timeit import default_timer as timer
from cython_naive import mandelbrot_naive

resolution = 500

RE_c = np.linspace(-2,1, resolution)
Im_c = np.linspace(1.5, -1.5, resolution)


MAX_iterations=100
threshold = 2

fracs = np.zeros((resolution, resolution))

def plot_brot(fractals):
    mplpp.imshow(fractals, cmap=mplpp.cm.hot)
    mplpp.title('Mandelbrot fractal')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_naive.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.close() 


n_i = 0
n_j = 0
r=0
co=0

threshold = 2

iterations = 100

matrix = np.zeros((resolution,resolution),dtype = complex)
start_timer = timer()
for i in RE_c:
    n_j=0
    r +=1
    co =0
    for j in Im_c:
        c = complex(i,j)
        co +=1 
        matrix[co-1,r-1] = c
        fracs[n_j,n_i] = mandelbrot(c, threshold, iterations)
        #print(c)
        #print(mandelbrot(c))
        n_j = n_j +1
    n_i = n_i +1
end_timer = timer()
time = end_timer - start_timer
def _plot(mfractal):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    
    
    
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' cython took: {time:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()
    
    
    print(f"cython took: {time:.7f}s")


_plot(fracs)
