import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import matplotlib.pyplot as plt
import time
from PIL import Image

# PAIRED ARRAY
b1 = np.ones((13,17))
b2 = np.ones((12,17))
res1_lo = np.argwhere(b1 == 1)
res2_lo = np.argwhere(b2 == 1)
res1_empty = np.array(
    [0,1,2,15,16,17,
     17*2-1,17*3-1,17*4-1,17*5-1,17*6-1,
     17*6,17*7-1,17*7,17*7+2,17*8-1,
     17*8,17*8+1,17*9,17*9+1,17*9+11,
     17*10,17*10+1,17*11,17*11+1,17*11+7,17*11+9,
     17*12-1,17*12,17*12+1,17*12+2,17*12+3,17*12+5,
     17*12+6,17*12+7,17*12+8,17*12+9,17*12+10,17*12+11,
     17*12+12,17*12+13,17*12+14,17*12+15,17*12+16])
res2_empty = np.array(
    [0,1,14,15,16,17,
     17*2-1,17*3-1,17*4-1,17*5-1,17*6-1,
     17*6,17*6+4,17*7-1,17*7,17*8-1,
     17*8,17*8+1,17*8+2,17*9,17*9+1,17*9+2,
     17*10,17*10+1,17*10+5,17*11-1,17*11,17*11+1,
     17*11+2,17*11+14,17*11+15])

res1_lo = np.delete(res1_lo, res1_empty, axis=0)
res2_lo = np.delete(res2_lo, res2_empty, axis=0)

res1 = np.array([np.array([yi*850, xi*600])+500 for yi, xi in res1_lo])
res2 = np.array([np.array([yi*850, xi*600])+500 for yi, xi in res2_lo])
im = np.zeros((11000, 11000), dtype = np.uint16)
for k in range(res1.shape[0]):
    y, x = res1[k][0], res1[k][1]
    im[y:y+200,x:x+200] = 255
for k in range(res2.shape[0]):
    y, x = res2[k][0], res2[k][1]
    im[y+250:y+450,x:x+200] = 255

cp.cuda.Device(1).use()

"""
pattern_dim = 40
threshold = 1/3
random_pattern = np.random.rand(pattern_dim, pattern_dim)
binary_pattern = np.where(random_pattern < threshold, 1, 0)

org_locs = np.argwhere(binary_pattern == 1)

org_locs_scaled = org_locs*200+25
centroids = []
im = np.zeros((8000, 8000), dtype = np.uint8)

for y, x in org_locs_scaled:
    im[y:y+150,x:x+150] = 255
    centroids.append((y+75, x+75))

#mat = cp.random.randn(160, 160)    
"""
sigma = 4

start_time = time.time()
im_blur = ndimage.gaussian_filter(cp.array(im), sigma, mode = 'constant')
im_blur_norm = im_blur * sigma * cp.sqrt(cp.pi)
im_sx = ndimage.sobel(im_blur_norm, axis=1, mode='constant')
im_sy = ndimage.sobel(im_blur_norm, axis=0, mode='constant')
im_sobel = cp.hypot(im_sx, im_sy)
im_lap = ndimage.gaussian_laplace(cp.array(im), sigma=sigma, mode='constant')
print("--- %s seconds ---" % (time.time() - start_time))

imSave = Image.fromarray(im_lap.get())
imSave = imSave.convert('RGB')
imSave.save("paired_lap_constant.png")

#plt.imshow(im_sobel)