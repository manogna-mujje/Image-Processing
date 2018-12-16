import numpy as np
from skimage import data, img_as_float
import skimage.filters
from skimage.restoration import (richardson_lucy)
from skimage import io
import os.path
import time
from mpi4py import MPI 
from numba import jit

curPath = os.path.abspath(os.path.curdir) 
noisyDir = os.path.join(curPath,'noisy') 
denoisedDir = os.path.join(curPath,'denoised')


@jit
def loop(imgFiles,rank):
    for f in imgFiles:
        img = img_as_float(data.load(os.path.join(noisyDir,f))) 
        psf = np.ones((5, 5)) / 25
        startTime = time.time()
        img = richardson_lucy(img, psf, iterations=30)
        io.imsave(os.path.join(denoisedDir,f), img)
        print ("Process %d: Took %f seconds for %s" %(rank, time.time() - startTime, f))

def parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    totalStartTime = time.time()
    numFiles = int(10/size)
    imgFiles = ["%.4d.jpg"%x for x in range(rank*numFiles+1, (rank+1)*numFiles+1)] 
    loop(imgFiles,rank)
    print("Total time %f seconds" %(time.time() - totalStartTime))

if __name__=='__main__': 
    parallel()