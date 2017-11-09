import numpy as np
import cv2

def dilation(img, kernel):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
        localmax = 0
        for item in kernel:
          x, y = item
          if (row + x) >= 0 and (row + x) < img.shape[0] and \
             (col + y) >= 0 and (col + y) < img.shape[1]:
               localmax = max(localmax, img[row + x, col + y])
          ret_img[row, col] = localmax
  return ret_img

def erosion(img, kernel):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      check = True
      localmin = np.inf
      for item in kernel:
        x, y = item
        if (row + x) > 0 and (row + x) < (img.shape[0] - 1) and \
           (col + y) > 0 and (col + y) < (img.shape[1] - 1):
             if (img[row + x, col + y] == 0):
               check = False
               break
             else:
               localmin = min(localmin, img[row+x, col+y])
        else:
          check = False
          break
      if check:
        ret_img[row, col] = localmin
  return ret_img

def opening(img, kernel):
  return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
  return erosion(dilation(img, kernel), kernel)

def main(): 
  # usage: python ./hw5.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # kernel is a 3-5-5-5-3 octagon
  # the origin is at the center
  kernel = [[-2, -1], [-2, 0], [-2, 1],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -1], [2, 0], [2, 1]]

  # Do dilation
  print 'do dilation for grayscale lena.bmp...\r\n'
  img_dil = dilation(img, kernel)
  cv2.imwrite('dil.bmp', img_dil)
  
  # Do erosion
  print 'do erosion for grayscale lena.bmp...\r\n'
  img_ero = erosion(img, kernel)
  cv2.imwrite('ero.bmp', img_ero)
  
  # Do opening
  print 'do opening for grayscale lena.bmp...\r\n'
  img_open = opening(img, kernel)
  cv2.imwrite('open.bmp', img_open)
  
  # Do closing
  print 'do closing for grayscale lena.bmp...\r\n'
  img_close = closing(img, kernel)
  cv2.imwrite('close.bmp', img_close)

  print 'All tasks are done.'

if __name__ == '__main__':
  main()
