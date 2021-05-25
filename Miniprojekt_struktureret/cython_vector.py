# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:41:21 2021

@author: Victor
"""
#https://www.learnpythonwithrune.org/numpy-compute-mandelbrot-set-by-vectorization/
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as mplpp
from cython_naive import mandelbrot_vector
   
def _plot(mfractal, xy_limits, time):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' cython vector took: {time:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()


def get_frac(xy_limits, xy_points, threshold, iterations):
    (x_min, x_max, y_min, y_max) = xy_limits
    (x_points, y_points) = xy_points
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_points, endpoint=True),
                       np.linspace(y_min, y_max, y_points, endpoint=True))

    c = x + 1j*y
    
    mfractal  = np.zeros(c.shape, dtype=np.float)
    for ix in range(mfractal.shape[0]):
           mfractal[ix,:] = mandelbrot_vector(c[ix,:], threshold, iterations)     
    return mfractal 
     
     
        
        
if __name__ == '__main__':

    threshold = 2
    iterations = 100
    resolution = int(500)
    
    
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    X_POINTS, Y_POINTS = resolution, resolution
    
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    xy_points = (X_POINTS, Y_POINTS)
    
    
    start = timer()
    mfractal = get_frac(xy_limits, xy_points, threshold, iterations)
    end = timer()
    time = end - start
    
    print(f"cython vector took: {time:.7f}s")
    
    _plot(mfractal, xy_limits, time)
