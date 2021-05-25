# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:13 2021

@author: Victor
"""

import matplotlib.pyplot as mplpp
import numpy as np
import multiprocessing as mp
from functools import partial
from timeit import default_timer as timer


def parallel_mandelbrot(xy_limits, xy_points, threshold, iterations,P):
    (x_min, x_max, y_min, y_max) = xy_limits
    (x_points, y_points) = xy_points
    
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_points, endpoint=True),
                       np.linspace(y_min, y_max, y_points, endpoint=True))
    c = x + 1j*y
    
    #pool of p processes
    pool = mp.Pool(processes=P)
   
    #http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    
    
    single_mpoint= partial(_mpoint, threshold = threshold, iterations = iterations)
    
    #results = pool.map_async(in_circle, rvals, chunksize=1)
    start_timer = timer()
    mfractal = pool.map(single_mpoint, c.flatten())
    mfractal = np.array(mfractal).reshape((c.shape))
    end_timer = timer()
    time = end_timer - start_timer

    return mfractal, time
  
    
def _plot(mfractal, xy_limits, time, p):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' multiprocessing took: {time:.5f}s'+ f"p is {p:.1f}")
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()

    
def _mpoint(c, threshold, iterations):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    
    """
    z = c.copy()
    #z = np.zeros(c.shape, dtype=complex)
    
    for iteration in range(1, iterations):
        z *= z;   z += c
        if np.abs(z) > threshold:
            return 100.0 * iteration / iterations
    return 100.0
    
# %% Main
if __name__ == '__main__':
    
    #set parameters 
    threshold = 2
    iterations = 100
    resolution = int(500)
    
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    
    X_POINTS, Y_POINTS = resolution, resolution  # Exe. time increases as O(X_POINTS*Y_POINTS)
    
    prange=1
    mfractal = np.zeros(prange)
    times = np.zeros(prange).astype(float)
    
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    xy_points = (X_POINTS, Y_POINTS)
    for i in range(prange): 
        P=i+1   
        
        mfractal, time = parallel_mandelbrot(xy_limits, xy_points, threshold, iterations,P)
        times[i] = time
        

        #_plot(mfractal, xy_limits, time, P)
    p_range = np.linspace(1, (prange*2)-1, prange) 
    
    mplpp.plot(p_range,times)
    mplpp.title("multiprocessing")
    mplpp.xlabel("processors")
    mplpp.ylabel("time")
    mplpp.show()
