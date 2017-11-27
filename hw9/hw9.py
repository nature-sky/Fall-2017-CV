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
  
def gradient_magnitude(neighbors, masks):
  magnitude = []
  num = len(masks)
  size_x = len(masks[0])
  size_y = len(masks[0][0])

  for i in xrange(num):
    r = 0
    for row in xrange(size_x):
      for col in xrange(size_y):
        r += neighbors[row][col] * masks[i][row][col]
    magnitude.append(math.pow(r, 2))
  return math.sqrt(sum(magnitude))

def max_magnitude(neighbors, masks):
  magnitude = []
  num = len(masks)
  size_x = len(masks[0])
  size_y = len(masks[0][0])

  for i in xrange(num):
    r = 0
    for row in xrange(size_x):
      for col in xrange(size_y):
        r += neighbors[row][col] * masks[i][row][col]
    magnitude.append(r)
  return max(magnitude)

def robert_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[-1, 0], [0, 1]], [[0, -1], [1, 0]]]
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = []
        neighbors.append([get_val(img, row, col), get_val(img, row, col+1)])
        neighbors.append([get_val(img, row+1, col), get_val(img, row+1, col+1)])
        if(gradient_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def prewitt_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[-1, -1, -1], [0, 0, 0], [1, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [3, 3])
        if(gradient_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def sobel_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [3, 3])
        if(gradient_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def frei_chen_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  sqrt2 = math.sqrt(2)
  masks = [[[-1, -sqrt2, -1], [0, 0, 0], [1, sqrt2, 1]], [[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [3, 3])
        if(gradient_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def kirsch_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], \
           [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], [[5, 5, -3], [5, 0, -3], [-3, -3, -3]], \
           [[5, -3, -3], [5, 0, -3], [5, -3, -3]], [[-3, -3, -3], [5, 0, -3], [5, 5, -3]], \
           [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [3, 3])
        if(max_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def robinson_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], \
           [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[2, 1, 0], [1, 0, -1], [0, -1, -2]], \
           [[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[0, -1, -2], [1, 0, -1], [2, 1, 0]], \
           [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [3, 3])
        if(max_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img

def nevatia_operator(img, threshold):
  ret_img = np.zeros(img.shape, int)
  masks = [[[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]], \
           [[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]], \
           [[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]], \
           [[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]], \
           [[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]], \
           [[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]]]
  neighbors = []
 
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        neighbors = get_neighbors(img, (row, col), [5, 5])
        if(max_magnitude(neighbors, masks) > threshold): 
          ret_img[row, col] = 0
        else:
          ret_img[row, col] = 255  
  return ret_img
  
def main():
  
  # usage: python ./hw9.py [operator] [threshold]
  # default threshold = 12
  threshold = '12'

  if(len(sys.argv) == 2 and sys.argv[1] == '-h'):
    print "usage: python ./hw9.py [-h] [operator] [threshold]"
    print "Options and argmuments:"
    print "-h: print this help message and exit"
    print "operator: robert, prewitt, sobel, frei, kirsch, robinson, nevatia"
    print "threshold: an integer for the operator"

  else:
    assert(len(sys.argv) == 3) 
    operator = sys.argv[1]
    threshold = sys.argv[2]
    assert operator == 'robert' or \
           operator == 'prewitt' or \
           operator == 'sobel' or \
           operator == 'frei'  or \
           operator == 'kirsch' or \
           operator == 'robinson' or \
           operator == 'nevatia'

    # img is a 512*512 array
    img = cv2.imread('lena.bmp', 0)

    # Do Robert's operator
    if operator == 'robert':
      img_robert = robert_operator(img, int(threshold))
      cv2.imwrite('robert' + threshold + '.bmp', img_robert)
  
    # Do Prewitt edge detector
    elif operator == 'prewitt':
      img_prewitt = prewitt_operator(img, int(threshold))
      cv2.imwrite('prewitt' + threshold + '.bmp', img_prewitt)

    # Do Sobel edge detector
    elif operator == 'sobel':
      img_sobel = sobel_operator(img, int(threshold))
      cv2.imwrite('sobel' + threshold + '.bmp', img_sobel)

    # Do Frei and Chen edge detector
    elif operator == 'frei':
      img_frei_chen = frei_chen_operator(img, int(threshold))
      cv2.imwrite('frei_chen' + threshold + '.bmp', img_frei_chen)

    # Do Kirsch's compass operator
    elif operator == 'kirsch':
      img_kirsch = kirsch_operator(img, int(threshold))
      cv2.imwrite('kirsch' + threshold + '.bmp', img_kirsch)

    # Do Robinson's compass operator
    elif operator == 'robinson':
      img_robinson = robinson_operator(img, int(threshold))
      cv2.imwrite('robinson' + threshold + '.bmp', img_robinson)

    # Do Nevatia-Babu 5x5 operator
    else:
      img_nevatia = nevatia_operator(img, int(threshold))
      cv2.imwrite('nevatia' + threshold + '.bmp', img_nevatia)

    print "The operator finished."
  
if __name__ == '__main__':
  main()
