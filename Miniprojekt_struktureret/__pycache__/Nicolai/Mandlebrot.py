# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:46:39 2021

@author: almsk
"""

## Fixed problemet på min bærbar med
# https://github.com/inducer/pyopencl/issues/442

# %% Imports 
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import multiprocessing as mp

import dask.multiprocessing
from dask.distributed import Client, wait
import dask

import pyopencl as cl

from numba import jit

# %% Mandlebrot Naive
def Mandlebrot_naive(lim, res_re, res_im, threshold, iterations):
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal  = np.zeros(c.shape, dtype=np.float)
    for ix in range(mfractal.shape[0]):
        for iy in range(mfractal.shape[1]):
            mfractal[ix, iy] = _mpoint_naive(c[ix, iy], threshold, iterations)
            
    return mfractal 

# %% Mandlebrot Numpy
def Mandlebrot_numpy(lim, res_re, res_im, threshold, iterations):
    x_min, x_max, y_min, y_max = lim      
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = (x + 1j*y)
    
    # Compute the Mandelbrot fractal
    mfractal  = np.zeros(c.shape, dtype=np.float)
    for ix in range(len(x)):
            mfractal[ix, :] = _mpoint_numpy(c[ix, :], threshold, iterations)
            
    return mfractal

# %% Mandlebrot Numba
def Mandlebrot_numba(lim, res_re, res_im, threshold, iterations):
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                       np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y

    # Compute and plot the Mandelbrot fractal
    mfractal  = np.zeros(c.shape, dtype=np.float)
    for ix in range(mfractal.shape[0]):
        for iy in range(mfractal.shape[1]):
            mfractal[ix, iy] = _mpoint_numba(c[ix, iy], threshold, iterations)
            
    return mfractal 


# %% Mandlebrot Multiprocessing with numpy
def Mandlebrot_multiprocessing(lim, res_re, res_im, threshold, iterations,p):
    x_min, x_max, y_min, y_max = lim
    # Form the complex grid
    x,y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                      np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y
        
    # Create a pool with p processes
    pool = mp.Pool(processes=p)
    
    # Take the _mpoint function and specify the two constants 
    _mpoint_p = partial(_mpoint_numpy, T=threshold, I=iterations)
    
    results = [pool.apply_async(_mpoint_p, [c[ix,:]]) for ix in range(len(x))]
    result = [result.get() for result in results]    
              
    return result


# %% Mandlebrot Dask - NOT DONE
def Mandlebrot_Dask(lim, res_re, res_im, threshold, iterations,p):
    x_min, x_max, y_min, y_max = lim 
    # Form the complex grid
    x,y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                      np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y
    
    # Compute and plot the Mandelbrot fractal
    mfractal  = np.zeros(c.shape, dtype=np.float)
    
    # Start the client with p workers
    client = Client(n_workers=p)
    
    # Take the _mpoint function and specify the two constants 
    _mpoint_p = partial(_mpoint_numpy, threshold=threshold,iterations=iterations)

    # Maps all the values from 'c' into the function _mpoint_p
    futures  = []
    for parameters in c.flatten():
        future = client.submit(_mpoint_p, parameters)
        futures.append(future)
    # Reshape the results 
    #mfractal = np.array().reshape((c.shape))

    mfractal = np.array(client.gather(futures)).reshape((c.shape))
    
    client.close()
     
    return mfractal   

    
# %% Mandlebrot GPU - with pyopencl
def Mandlebrot_GPU(lim, res_re, res_im, threshold, iterations):
    x_min, x_max, y_min, y_max = lim      
    # Form the complex grid
    x,y = np.meshgrid(np.linspace(x_min, x_max, res_re, endpoint=True),
                      np.linspace(y_min, y_max, res_im, endpoint=True))
    c = x + 1j*y
    
    # Compute the Mandelbrot fractal
    mfractal  = np.zeros([res_re, res_im], dtype=np.float64)
        
    mfractal = _mpoint_opencl(c.astype(np.complex128), threshold, iterations)
              
    return mfractal


# %% mpoint
def _mpoint_numpy(c, T = 2, I = 100):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    
    This version take care of overflow
    
    """
    z = np.zeros(c.shape, dtype=complex)
    Iota = np.full(z.shape, I)

    for i in range(1, I+1):
        z[Iota > (i-1)] = np.square(z[Iota > (i-1)]) + c[Iota > (i-1)]   # Only calculated for the ones which hasn't diverged 
        
        Iota[np.logical_and(np.abs(z) > T, Iota == I)] = i # Write iteration number to the matrix 

    return Iota

def _mpoint_naive(c, T, I):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    Does not works with arrays. 
    
    """
    z = 0;
    for i in range(1, I):
        z = z**2 + c
        
        if np.abs(z) > T:
            return i
    return 100


@jit
def _mpoint_numba(c, T, I):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    Does not works with arrays. 
    
    """
    z = 0;
    for i in range(1, I):
        z = z**2 + c
        
        if np.abs(z) > T:
            return i
    return 100

def _mpoint_opencl(c, T, I):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    output = np.empty(c.shape, dtype=np.int32)

    mf = cl.mem_flags
    c_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg = cl.Program(
    ctx,
    """
    #define PYOPENCL_DEFINE_CDOUBLE
    #include <pyopencl-complex.h>
    __kernel void mandelbrot
             (
             __global const cdouble_t *c,
             __global int *output,
             ushort const T,
             ushort const I,
             int dim
             )
    {
        int idx = get_global_id(0);
        int idy = get_global_id(1);
        
        cdouble_t z = cdouble_new(0,0);

        
        int i = 0;
        while(i < I & cdouble_abs(z) <= T)
        {
            z = cdouble_mul(z,z);
            z = cdouble_add(z,c[idy*dim + idx]);   
            i = i + 1;
            
        }
         output[idy*dim + idx] = i;       
    }
    """,
    ).build()

    prg.mandelbrot(
        queue, output.shape, None, c_opencl, output_opencl, np.uint16(T), np.uint16(I), np.int32(c.shape[0])
    )

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

# %% plot
def _plot(mfractal, lim, title):
    """
    Plot the mandelbrot set contained in the mfractal matrix.
    
    """
    x_min, x_max, y_min, y_max = lim    
    
    # Make plot and save figure
    plt.imshow(np.log(mfractal), cmap=plt.cm.hot, extent=[x_min, x_max, y_min, y_max])
    plt.title(title)
    plt.xlabel('Re[c]')
    plt.ylabel('Im[c]')
    plt.show()
    plt.savefig(f"{title}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()



# %% Main
if __name__ == '__main__':
    # Number of processes
    p = 8
    
    # Constants - Limits
    lim = [-2, 1, -1.5, 1.5] # [x_min, x_max, y_min, y_max]

    # Constants - Resolution
    p_re, p_im = [500, 500]

    # Constants - Threshold
    T = 2

    # Constants - Number of Iterations
    iterations = 100
      
    # ---- Naive ----
    """
    # Start timer
    t_start = timer()

    # calculate the fractals 
    mfractal_naive = Mandlebrot_naive(lim=lim, res_re=p_re, res_im=p_im, threshold=T, iterations=iterations)

    # Stop timer
    t_stop = timer()
    t_naive = t_stop - t_start

    # Plot
    _plot(mfractal_naive, lim, 'Mandlebrot Naive')

    print(f"Mandlebrot Naive took {t_naive}s")
    """

    # ---- Numba ----
    
    # Start timer
    t_start = timer()

    # calculate the fractals       
    mfractal_numba = Mandlebrot_numba(lim=lim, res_re=p_re, res_im=p_im, threshold=T, iterations=iterations)  
    
    # Stop timer
    t_stop = timer()
    t_numba = t_stop - t_start

    # Plot
    _plot(mfractal_numba, lim, 'Mandlebrot Numba')      

    print(f"Mandlebrot Numba took {t_numba}s")
    
    
    
    # ---- Numpy ----
    
    # Start timer
    t_start = timer()

    # calculate the fractals       
    mfractal_numpy = Mandlebrot_numpy(lim=lim, res_re=p_re, res_im=p_im, threshold=T, iterations=iterations)  
    
    # Stop timer
    t_stop = timer()
    t_numpy = t_stop - t_start

    # Plot
    _plot(mfractal_numpy, lim, 'Mandlebrot Numpy')      

    print(f"Mandlebrot Numpy took {t_numpy}s")
    
    
    # ---- Multiprocessing ----
    """
    # Start timer
    t_start = timer()
    
    # calculate the fractals     
    mfractal_multiprocessing = Mandlebrot_multiprocessing(lim=lim, res_re=p_re, res_im=p_im, threshold=T, iterations=iterations, p=p) 

    # Stop timer
    t_stop = timer()
    t_multiprocessing = t_stop - t_start

    # Plot
    _plot(mfractal_multiprocessing, lim, 'Mandlebrot Multiprocessing')

    print(f"Mandlebrot Multiprocessing took {t_multiprocessing}s")    
    """
    
    # ---- GPU ----
    # Start timer
    t_start = timer()
    
    # calculate the fractals     
    mfractal_GPU = Mandlebrot_GPU(lim=lim, res_re=p_re, res_im=p_im, threshold=T, iterations=iterations) 

    # Stop timer
    t_stop = timer()
    t_GPU = t_stop - t_start

    # Plot
    _plot(mfractal_GPU, lim, 'Mandlebrot GPU')

    print(f"Mandlebrot GPU took {t_GPU}s")   
  
    
