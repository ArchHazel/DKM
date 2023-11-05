import numpy as np
import cv2

im_Mask = np.zeros((10, 8), dtype=np.uint16)
print(im_Mask.shape)

im_Mask[2:5] = 300
im_Mask[6:7] = 10

np.save('data/im_mask.npy', im_Mask)
impath = 'data/test_truncated.png'
cv2.imwrite(impath, im_Mask)

im2 = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
print(im2.shape)
print(im2)

im2_np = np.load('data/im_mask.npy')
print(im2_np)