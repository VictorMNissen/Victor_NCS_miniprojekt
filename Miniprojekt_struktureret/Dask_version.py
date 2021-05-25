# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:13 2021

@author: Victor
"""

import matplotlib.pyplot as mplpp
import numpy as np
import time
from functools import partial
import dask
import dask.multiprocessing
from dask.distributed import Client, wait

def dask_mandelbrot(xy_limits, xy_points, threshold, iterations,P):
    (x_min, x_max, y_min, y_max) = xy_limits
    (x_points, y_points) = xy_points
    
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_points, endpoint=True),
                       np.linspace(y_min, y_max, y_points, endpoint=True))
    c = x + 1j*y
    
    mfractal  = np.zeros(c.shape, dtype=np.float) # nok ikke nødvendig
    
    #pool of p processes
    
    client = Client(n_workers = P)
    #http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    
    
    single_mpoint= partial(_mpoint, threshold = threshold, iterations = iterations)
    
    
    start = time.time()
    
    
    futures  = []
    #for parameters in c.flatten():
    #    future = client.submit(single_mpoint, parameters)
    #    futures.append(future)
    
    for ix in range(len(x)):
        future = client.submit(single_mpoint,c[ix,:])
        futures.append(future)
    
    # Reshape the results 
    #mfractal = np.array(client.gather(futures)).reshape((c.shape))
    
    mfractal = client.gather(futures)
    
    stop = time.time()
    time_ex = stop-start
    client.close()
     
    return mfractal, time_ex   
    
      
        
def _plot(mfractal, xy_limits, time_ex):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot fractal'+ f' dask took: {time_ex:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()

    
def _mpoint(c, threshold, iterations):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    
    """
    z = np.zeros(c.shape, dtype=complex)
    frac = np.full(z.shape, iterations)

    for i in range(1, iterations+1):
        z[frac > (i-1)] = np.square(z[frac > (i-1)]) + c[frac > (i-1)]   # Only calculated for the ones which hasn't diverged 
        
        frac[np.logical_and(np.abs(z) > threshold, frac == iterations)] = i # Write iteration number to the matrix 

    return frac

# %% Main
if __name__ == '__main__':
    
    #set parameters 
    P=4
    threshold = 2
    iterations = 100
    resolution = int(500)
    
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    
    X_POINTS, Y_POINTS = resolution, resolution  # Exe. time increases as O(X_POINTS*Y_POINTS)
    
    
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    xy_points = (X_POINTS, Y_POINTS)
     
    mfractal, time_ex = dask_mandelbrot(xy_limits, xy_points, threshold, iterations,P)
    
    
    
    _plot(mfractal, xy_limits, time_ex)
   
    
    
    
