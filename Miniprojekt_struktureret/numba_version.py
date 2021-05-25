# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:13 2021

@author: Victor
"""

import matplotlib.pyplot as mplpp
import numpy as np
from numba import jit
from timeit import default_timer as timer

@jit(nopython=True)
def mandelbrot(c, threshold, iterations):

    z = 0
    for iteration in range(1, iterations):
        z *= z;   z += c
        if np.abs(z) > threshold:
            return 100.0 * iteration / iterations
    return 100.0
    

def plot_brot(fractals):
    mplpp.imshow(fractals, cmap=mplpp.cm.hot)
    mplpp.title('Mandelbrot fractal')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_naive.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.close() 


def numba_mandelbrot(xy_limits, xy_points, threshold, iterations):
    (x_min, x_max, y_min, y_max) = xy_limits
    (x_points, y_points) = xy_points
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_points, endpoint=True),
                       np.linspace(y_min, y_max, y_points, endpoint=True))

    c = x + 1j*y
    mfractal  = np.zeros(c.shape, dtype=np.float)
    
    for ix in range(mfractal.shape[0]):
        for iy in range(mfractal.shape[1]):
            mfractal[ix, iy] = mandelbrot(c[ix, iy], threshold, iterations)
    return mfractal 



def _plot(mfractal, time):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' numba took: {time:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()
    
    
    
if __name__ == '__main__':

    resolution = 500
    
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    
    X_POINTS, Y_POINTS = resolution, resolution
    xy_points = (X_POINTS, Y_POINTS)
    
    iterations=100
    threshold = 2
    start_time = timer()
    fracs = numba_mandelbrot(xy_limits, xy_points, threshold, iterations)
    end_time = timer()
    time = end_time - start_time
    
    _plot(fracs,time)
    
    
    