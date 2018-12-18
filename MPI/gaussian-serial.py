import numpy as np
from skimage import data, img_as_float, img_as_uint 
from scipy.signal import convolve2d as conv2
import skimage.filters
from skimage import io, color
import os.path
import time
import scipy.misc
from skimage.filters import (gaussian)

curPath = os.path.abspath(os.path.curdir) 
inputDir = os.path.join(curPath,'input') 
resultDir = os.path.join(curPath,'output')

def loop(imgFiles):
    for f in imgFiles:
        img = data.load(os.path.join(inputDir,f))
        startTime = time.time()
        img = gaussian(img, sigma=1, multichannel=True)
        io.imsave(os.path.join(resultDir,f), img)
        print("Took %f seconds for %s" %(time.time() - startTime, f))

def serial():
    total_start_time = time.time()
    # imgFiles = ["%.4d.png"%x for x in range(1,2)]
    imgFiles = ["%.4d.jpg"%x for x in range(1,11)]
    loop(imgFiles)
    print("Total time %f seconds" %(time.time() - total_start_time))

if __name__=='__main__': 
    serial()


