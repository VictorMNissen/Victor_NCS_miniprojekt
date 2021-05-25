# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:13 2021

@author: Victor
"""



import matplotlib.pyplot as mplpp
import numpy as np
from timeit import default_timer as timer





def mandelbrot(c, threshold, iterations):
    
    z = 0
    n = 0 
    while(abs(z) <= threshold and n < iterations): 
        z *= z; z +=c
        n = n+1 
        
    return n*100/iterations
    """
    z = 0
    for iteration in range(1, iterations):
        z *= z;   z += c
        if np.abs(z) > threshold:
            return 100.0 * iteration / iterations
    return 100.0
    """


def plot_brot(fractals):
    """
    Plotting the mandelbrot fractals 
    
    Parameters
    ----------
    fractals : matrix containing the fractals of the mandelbrot function
        DESCRIPTION.

    Returns
    -------
    None.

    """
    mplpp.imshow(fractals, cmap=mplpp.cm.hot)
    mplpp.title('Mandelbrot fractal')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_naive.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.close() 



def calc_frac(Im_c, RE_c, threshold, iterations):
    n_i = 0
    n_j = 0
    r=0
    co=0
    fracs = np.zeros((resolution, resolution))
    
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
    return fracs, time
def _plot(mfractal, time):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    xy_limits = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    
    
    
    # Make plot and save figure
    mplpp.imshow(np.log(mfractal), cmap=mplpp.cm.hot, extent=xy_limits)
    mplpp.title('Mandelbrot' + f' naive took: {time:.5f}s')
    mplpp.xlabel('Re[c]')
    mplpp.ylabel('Im[c]')
    mplpp.savefig('mfractal_map_multi.pdf', bbox_inches='tight', pad_inches=0.05)
    mplpp.show()
    
    
    print(f"naive took: {time:.7f}s")
if __name__ == '__main__':
    resolution = 500
    
    RE_c = np.linspace(-2,1, resolution)
    Im_c = np.linspace(1.5, -1.5, resolution)
    threshold = 2
    iterations = 100;
    
    fracs, time = calc_frac(Im_c,RE_c,threshold,iterations)
    
    
    
    
    
    _plot(fracs,time)

    