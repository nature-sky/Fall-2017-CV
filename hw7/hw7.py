import numpy as np
import cv2

threshold = 128

def binarize(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      if img[row, col] >= threshold:
        ret_img[row, col] = 255
      else:
        ret_img[row, col] = 0  
  return ret_img
  
def downsample(img, blocksize):
  width = img.shape[0] / blocksize
  height = img.shape[1] / blocksize
  ret_img = np.zeros((width, height), int)
  for row in xrange(width):
    for col in xrange(height):
      ret_img[row, col] = img[row * blocksize, col * blocksize]
  return ret_img

def get_val(img, x, y):
  if x < 0 or x > img.shape[0]-1 or y < 0 or y > img.shape[1]-1:
    return 0
  return img[x, y]

def h_func_ib(c, d):
  if c == d:
    return c
  elif c != d:
    return 'b'

def f_func_ib(c):
  if c == 'b':
    return 2
  elif c != 'b':
    return 1   

# for 4-connected
def interior_border(img_origin, neighbors):
  # 0: background pixel
  # 1: interior pixel
  # 2: border pixel
  ret_img = np.zeros(img_origin.shape, int)
  for row in xrange(img_origin.shape[0]):
    for col in xrange(img_origin.shape[1]):
      # foreground pixel
      if img_origin[row, col] > 0:
        x0 = img_origin[row, col]
        x1 = get_val(img_origin, row+neighbors[1][0], col+neighbors[1][1])
        x2 = get_val(img_origin, row+neighbors[2][0], col+neighbors[2][1])
        x3 = get_val(img_origin, row+neighbors[3][0], col+neighbors[3][1])
        x4 = get_val(img_origin, row+neighbors[4][0], col+neighbors[4][1])
       
        a0 = x0
        a1 = h_func_ib(a0, x1)
        a2 = h_func_ib(a1, x2)
        a3 = h_func_ib(a2, x3)
        a4 = h_func_ib(a3, x4)
        ret_img[row, col] = f_func_ib(a4)
  return ret_img 

def h_func_pr(a, m):
  if a == m:
    return 1
  else:
    return 0

# for 4-connected
def pair_relation(img_int_bor, neighbors):
  # 0: background pixel 
  # 1: p 
  # 2: q 
  ret_img = np.zeros(img_int_bor.shape, int)
  for row in xrange(img_int_bor.shape[0]):
    for col in xrange(img_int_bor.shape[1]):
      # foreground pixel
      if img_int_bor[row, col] > 0: 
        l = 2 # border pixel
        m = 1 # interior pixel
        theta = 1
        
        x0 = img_int_bor[row, col]
        x1 = get_val(img_int_bor, row+neighbors[1][0], col+neighbors[1][1])
        x2 = get_val(img_int_bor, row+neighbors[2][0], col+neighbors[2][1])
        x3 = get_val(img_int_bor, row+neighbors[3][0], col+neighbors[3][1])
        x4 = get_val(img_int_bor, row+neighbors[4][0], col+neighbors[4][1])
       
        h1 = h_func_pr(x1, m)
        h2 = h_func_pr(x2, m)
        h3 = h_func_pr(x3, m)
        h4 = h_func_pr(x4, m)
        
        h = [h1, h2, h3, h4]
        if (sum(h) >= theta) and x0 == l:
          ret_img[row, col] = 1
        else:
          ret_img[row, col] = 2
  return ret_img 

def h_func_cs(b, c, d, e):
  if b == c and (b != d or b != e):
    return 1
  else:
    return 0

def f_func_cs(a1, a2, a3, a4, x0):
  arr = [a1, a2, a3, a4]
  if (sum(arr) == 1):
    return 1
  else:
    return 0   

def connected_shrink(img_origin, neighbors):
  # 0: background pixel or unremovable foreground pixel
  # 1: removable foreground pixel
  ret_img = np.zeros(img_origin.shape, int)
  for row in xrange(img_origin.shape[0]):
    for col in xrange(img_origin.shape[1]):
      # foreground pixel
      if img_origin[row, col] > 0: 
        x0 = img_origin[row, col]
        x1 = get_val(img_origin, row+neighbors[1][0], col+neighbors[1][1])
        x2 = get_val(img_origin, row+neighbors[2][0], col+neighbors[2][1])
        x3 = get_val(img_origin, row+neighbors[3][0], col+neighbors[3][1])
        x4 = get_val(img_origin, row+neighbors[4][0], col+neighbors[4][1])
        x5 = get_val(img_origin, row+neighbors[5][0], col+neighbors[5][1])
        x6 = get_val(img_origin, row+neighbors[6][0], col+neighbors[6][1])
        x7 = get_val(img_origin, row+neighbors[7][0], col+neighbors[7][1])
        x8 = get_val(img_origin, row+neighbors[8][0], col+neighbors[8][1])
      
        a1 = h_func_cs(x0, x1, x6, x2)
        a2 = h_func_cs(x0, x2, x7, x3)
        a3 = h_func_cs(x0, x3, x8, x4)
        a4 = h_func_cs(x0, x4, x5, x1)
        ret_img[row, col] = f_func_cs(a1, a2, a3, a4, x0)
  return ret_img

def h_func_Yokoi(b, c, d, e):
  if b == c and (d != b or e != b):
    return 'q'
  elif b == c and (d == b and e == b):
    return 'r'
  elif b != c:
    return 's'

def f_func_Yokoi(a1, a2, a3, a4):
  arr = [a1, a2, a3, a4]
  if (arr.count('r') == len(arr)):
    return 5
  else:
    return arr.count('q')   

def Yokoi(img, neighbors):
  ret_img = np.empty(img.shape, int)
  ret_img.fill(-1) # Initialize value with -1. It means background pixels.
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      if img[row, col] > 0:
        x0 = img[row, col]
        x1 = get_val(img, row+neighbors[1][0], col+neighbors[1][1])
        x2 = get_val(img, row+neighbors[2][0], col+neighbors[2][1])
        x3 = get_val(img, row+neighbors[3][0], col+neighbors[3][1])
        x4 = get_val(img, row+neighbors[4][0], col+neighbors[4][1])
        x5 = get_val(img, row+neighbors[5][0], col+neighbors[5][1])
        x6 = get_val(img, row+neighbors[6][0], col+neighbors[6][1])
        x7 = get_val(img, row+neighbors[7][0], col+neighbors[7][1])
        x8 = get_val(img, row+neighbors[8][0], col+neighbors[8][1])
        
        a1 = h_func_Yokoi(x0, x1, x6, x2)
        a2 = h_func_Yokoi(x0, x2, x7, x3)
        a3 = h_func_Yokoi(x0, x3, x8, x4)
        a4 = h_func_Yokoi(x0, x4, x5, x1)
        ret_img[row, col] = f_func_Yokoi(a1, a2, a3, a4)
  return ret_img 

def thin(img, neighbors):
  img_origin = np.copy(img)
  changed = False
  while True:
    changed = False
    img_origin_old = np.copy(img_origin)
 
    # Step 1
    img_int_bor = interior_border(img_origin, neighbors)  
    
    # Step 2
    img_mark = pair_relation(img_int_bor, neighbors)
    
    # Step 3
    #img_Yokoi = Yokoi(img_origin, neighbors)
    #img_removable = (img_Yokoi == 1) * 1
    img_removable = connected_shrink(img_origin, neighbors)
    for row in xrange(img_origin.shape[0]):
      for col in xrange(img_origin.shape[1]):
        if img_removable[row, col] == 1 and img_mark[row, col] == 1:
          img_origin[row, col] = 0
    
    for row in xrange(img_origin.shape[0]):
      for col in xrange(img_origin.shape[1]):
        if img_origin[row, col] != img_origin_old[row, col]:
          changed = True
          break

    if changed == False:
      break
  return img_origin


def main():
  
  # usage: python ./hw7.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # Do binarize
  img_bin = binarize(img)
  cv2.imwrite('binary.bmp', img_bin)
  
  # Downsample into 64*64 image(8 is the block size).
  img_down = downsample(img_bin, 8)
  cv2.imwrite('down_binary.bmp', img_down)

  # 8-connected neighborhood
  neighbors = [[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]

  # Do thinning for downsampled image.
  img_thin = thin(img_down, neighbors)
  print "thin operator finished."
  
  cv2.imwrite('thin_binary.bmp', img_thin)
  with open("thin.txt", "w") as txt_file:
    for row in xrange(img_thin.shape[0]):
      for col in xrange(img_thin.shape[1]):
        if (img_thin[row, col] > 0):
          txt_file.write("*")
        else:
          txt_file.write(" ")
      txt_file.write('\n')

if __name__ == '__main__':
  main()
