import numpy as np
import cv2

threshold = 128

def binarize(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      if img[row][col] >= threshold:
        ret_img[row][col] = 255
      else:
        ret_img[row][col] = 0  
  return ret_img

def dilation(img, kernel):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      if img[row, col] == 255:
        for item in kernel:
          x, y = item
          if (row + x) >= 0 and (row + x) < img.shape[0] and \
             (col + y) >= 0 and (col + y) < img.shape[1]:
               ret_img[row + x, col + y] = 255
  return ret_img

def erosion(img, kernel):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      check = True
      for item in kernel:
        x, y = item
        if (row + x) < 0 or (row + x) > (img.shape[0] - 1) or \
           (col + y) < 0 or (col + y) > (img.shape[1] - 1) or \
           img[row + x, col + y] == 0:
             check = False
             break
      if check:
        ret_img[row, col] = 255
  return ret_img

def opening(img, kernel):
  return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
  return erosion(dilation(img, kernel), kernel)

def complement(img):
  img_comp = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      if img[row, col] == 255:
        img_comp[row, col] = 0
      else:
        img_comp[row, col] = 255
  return img_comp

def intersect(img1, img2):
  img_inter = np.zeros(img1.shape, int)
  for row in xrange(img1.shape[0]):
    for col in xrange(img1.shape[1]):
      if img1[row, col] == 255 and img2[row, col] == 255:
        img_inter[row, col] = 255
      else:
        img_inter[row, col] = 0
  return img_inter

def hit_and_miss(img, j_kernel, k_kernel):
  return intersect(erosion(img, j_kernel), erosion(complement(img), k_kernel))

def main(): 
  # usage: python ./hw4.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # kernel is a 3-5-5-5-3 octagon
  # the origin is at the center
  kernel = [[-2, -1], [-2, 0], [-2, 1],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -1], [2, 0], [2, 1]]

  # Do binarize
  img_bin = binarize(img)
  cv2.imwrite('binary.bmp', img_bin)
 
  # Do dilation
  print 'do dilation for binary.bmp...\r\n'
  img_dil = dilation(img_bin, kernel)
  cv2.imwrite('bin_dil.bmp', img_dil)
  
  # Do erosion
  print 'do erosion for binary.bmp...\r\n'
  img_ero = erosion(img_bin, kernel)
  cv2.imwrite('bin_ero.bmp', img_ero)
  
  # Do opening
  print 'do opening for binary.bmp...\r\n'
  img_open = opening(img_bin, kernel)
  cv2.imwrite('bin_open.bmp', img_open)
  
  # Do closing
  print 'do closing for binary.bmp...\r\n'
  img_close = closing(img_bin, kernel)
  cv2.imwrite('bin_close.bmp', img_close)

  j_kernel = [[0, -1], [0, 0], [1, 0]]
  k_kernel = [[-1, 0], [-1, 1], [0, 1]]

  # Do hit and miss 
  print 'do hit and miss for binary.bmp...\r\n'
  img_ham = hit_and_miss(img_bin, j_kernel, k_kernel)
  cv2.imwrite('bin_ham.bmp', img_ham)

  print 'All tasks are done.'

if __name__ == '__main__':
  main()
