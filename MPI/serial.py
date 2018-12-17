import numpy as np
from skimage import data, img_as_float, img_as_uint 
from scipy.signal import convolve2d as conv2
import skimage.filters
from skimage import io, color
import os.path
import time
import scipy.misc
from skimage.restoration import (richardson_lucy)

curPath = os.path.abspath(os.path.curdir) 
originalDir = os.path.join(curPath,'original') 
noisyDir = os.path.join(curPath, 'noisy')
denoisedDir = os.path.join(curPath,'denoised')
def loop(imgFiles):
    for f in imgFiles:
        img = data.load(os.path.join(originalDir,f))
        bwimg = color.rgb2gray(img)
        # print(type(img))
        psf = np.ones((5, 5)) / 25
        img = conv2(bwimg, psf, 'same')

        # Add Noise to Image
        img = img.copy()
        img += (np.random.poisson(lam=25, size=img.shape) - 10) / 255.
        # io.imsave(os.path.join(noisyDir,f), img)
        scipy.misc.imsave(os.path.join(noisyDir,f), img)

        startTime = time.time()
        img = richardson_lucy(img, psf, iterations=30, clip=True)
        io.imsave(os.path.join(denoisedDir,f), img)
        print("Took %f seconds for %s" %(time.time() - startTime, f))

def serial():
    total_start_time = time.time()
    # imgFiles = ["%.4d.png"%x for x in range(1,2)]
    imgFiles = ["%.4d.jpg"%x for x in range(1,11)]
    loop(imgFiles)
    print("Total time %f seconds" %(time.time() - total_start_time))

if __name__=='__main__': 
    serial()