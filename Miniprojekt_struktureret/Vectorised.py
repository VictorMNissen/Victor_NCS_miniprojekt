# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:41:21 2021

@author: Victor
"""
"""
This file calculates the mandelbrot fractals using a vectorised method and plots it
"""

#https://www.learnpythonwithrune.org/numpy-compute-mandelbrot-set-by-vectorization/
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as mplpp
   
def _plot(mfractal, xy_limits, time):
    """
    Plotting the mandelbrot fractals

    Parameters
    ----------
    mfractal : matrix (float)
        Containing the fractals of the mandelbrot
    xy_limits : np.array (float)
        array containing the 2 limits for each axis 
        xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX) 
    time : float
        Used to print the time taken for the mandelbrot computation
    Returns
    -------
    Plot of the mandelbrot with execution time
    """

    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' vector took: {time:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()


def get_frac(xy_limits, xy_points, threshold, iterations):
    """
    

    Parameters
    ----------
    xy_limits : TYPE
        DESCRIPTION.
    xy_points : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.

    Returns
    -------
    mfractal : TYPE
        DESCRIPTION.

    """
    
    (x_min, x_max, y_min, y_max) = xy_limits
    (x_points, y_points) = xy_points
    x, y = np.meshgrid(np.linspace(x_min, x_max, x_points, endpoint=True),
                       np.linspace(y_min, y_max, y_points, endpoint=True))

    c = x + 1j*y
    
    mfractal  = np.zeros(c.shape, dtype=np.float)
    for ix in range(mfractal.shape[0]):
           mfractal[ix,:] = mandelbrot_vector(c[ix,:], threshold, iterations)     
    return mfractal 
      

def mandelbrot_vector(c, threshold, iterations):
    """
    

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.

    Returns
    -------
    taken : TYPE
        DESCRIPTION.

    """ 
    z = np.zeros(c.shape, dtype = complex)
    taken = np.full(z.shape, iterations)
    for i in range(1,iterations):
        z[taken > (i-1)] = np.square(z[taken > (i-1)]) + c[taken > (i-1)]   # Only calculated for the ones which hasn't diverged 
        
        taken[np.logical_and(np.abs(z) > threshold, taken == iterations)] = i # Write iteration number to the matrix 
    return taken

                #np.where(abs(z[:,i]) > threshold, z[:,i], 100.0 * i / iterations)
        # for j in range(rows):
    #     m = np.full((rows,1), True, dtype=bool)
    #     z = np.zeros((len(c),1),dtype=complex)    
    #     for i in range(1,iterations):
    #         z[m]*=z[m]
            
    #         z[m]+=c[m[:,0],j]
    #         frac = np.greater(np.abs(z), threshold, out=np.full((rows,1), False), where=m)
    #         fracs[frac[:,0],j] = (100.0 * i / iterations)     
    #         m[np.abs(z) > threshold] = False
    #         if (i == iterations-1):
    #             #fracs[fracs[:,j] > 1,j] -=1 ;
    #             fracs[fracs[:,j] == 0,j] = 100;     
        
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
    
    print(f"vector took: {time:.7f}s")
    
    _plot(mfractal, xy_limits, time)
    