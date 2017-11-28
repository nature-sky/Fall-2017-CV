import numpy as np
import cv2, math, sys

def get_val(img, x, y):
  if x < 0 or x > img.shape[0]-1 or y < 0 or y > img.shape[1]-1:
    return 0
  return img[x, y]

def get_neighbors(img, origin, sizes):
  neighbors = np.zeros((sizes[0], sizes[1]), int)
  half = sizes[0] / 2
  x = origin[0]
  y = origin[1]

  for row in xrange(-half, half + 1):
    for col in xrange(-half, half + 1):
      neighbors[half + row, half + col] = get_val(img, x + row, y + col)
  return neighbors
  
def magnitude(neighbors, mask):
  magnitude = 0
  size_x = len(mask)
  size_y = len(mask[0])

  for row in xrange(size_x):
    for col in xrange(size_y):
        magnitude += neighbors[row][col] * mask[row][col]
  return magnitude

def laplace(img, mask, mask_ratio, threshold):
  ret_img = np.zeros(img.shape, int)
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      neighbors = []
      neighbors = get_neighbors(img, (row, col), [3, 3])
      if((magnitude(neighbors, mask) / mask_ratio) > threshold): 
        ret_img[row, col] = 0
      else:
        ret_img[row, col] = 255
  return ret_img

def MVL(img, threshold):
  ret_img = np.zeros(img.shape, int)
  mask = [[2, -1, 2], [-1, -4, -1], [2, -1, 2]]
  mask_div_ratio = 3
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      neighbors = get_neighbors(img, (row, col), [3, 3])
      if((magnitude(neighbors, mask) / mask_div_ratio) > threshold): 
        ret_img[row, col] = 0
      else:
        ret_img[row, col] = 255 
  return ret_img

def LOG(img, threshold):
  ret_img = np.zeros(img.shape, int)
  mask = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
	  [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
	  [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
	  [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
	  [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
          [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
	  [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
	  [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
          [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
          [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
          [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      neighbors = get_neighbors(img, (row, col), [11, 11])
      if(magnitude(neighbors, mask) > threshold): 
        ret_img[row, col] = 0
      else:
        ret_img[row, col] = 255 
  return ret_img

def DOG(img, threshold):
  ret_img = np.zeros(img.shape, int)
  mask = [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
          [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
          [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
          [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
          [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
          [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
          [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
          [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
          [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
          [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
          [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      neighbors = get_neighbors(img, (row, col), [11, 11])
      if(magnitude(neighbors, mask) > threshold): 
        ret_img[row, col] = 255
      else:
        ret_img[row, col] = 0  
  return ret_img
  
def main():
  
  # usage: python ./hw10.py [operator] [threshold]
  # default threshold = 12
  threshold = '12'

  if(len(sys.argv) == 2 and sys.argv[1] == '-h'):
    print "usage: python ./hw10.py [-h] [operator] [threshold]"
    print "Options and argmuments:"
    print "-h: print this help message and exit"
    print "operator: laplace1, laplace2, MVL, LOG, DOG"
    print "threshold: an integer for the operator"

  else:
    assert(len(sys.argv) == 3) 
    operator = sys.argv[1]
    threshold = sys.argv[2]
    assert operator == 'laplace1' or \
           operator == 'laplace2' or \
           operator == 'MVL' or \
           operator == 'LOG' or \
           operator == 'DOG' 

    # img is a 512*512 array
    img = cv2.imread('lena.bmp', 0)

    # Do Laplace mask [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    if operator == 'laplace1':
      laplace_mask1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
      mask_div_ratio1 = 1
      img_laplace1 = laplace(img, laplace_mask1, mask_div_ratio1, int(threshold))
      cv2.imwrite('laplace1_' + threshold + '.bmp', img_laplace1)
  
    # Do Laplace mask [[1, 1, 1], [1, -8, 1], [1, 1, 1]] / 3
    if operator == 'laplace2':
      laplace_mask2 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
      mask_div_ratio2 = 3
      img_laplace2 = laplace(img, laplace_mask2, mask_div_ratio2, int(threshold))
      cv2.imwrite('laplace2_' + threshold + '.bmp', img_laplace2)

    # Do Minimum variance Laplacian
    elif operator == 'MVL':
      img_mvl = MVL(img, int(threshold))
      cv2.imwrite('MVL' + threshold + '.bmp', img_mvl)

    # Do Laplace of Gaussian
    elif operator == 'LOG':
      img_log = LOG(img, int(threshold))
      cv2.imwrite('LOG' + threshold + '.bmp', img_log)

    # Do Difference of Gaussian
    elif operator == 'DOG':
      img_dog = DOG(img, int(threshold))
      cv2.imwrite('DOG' + threshold + '.bmp', img_dog)

    print "The operator finished."
  
if __name__ == '__main__':
  main()
