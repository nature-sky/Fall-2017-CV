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

def h_func(b, c, d, e):
  if b == c and (d != b or e != b):
    return 'q'
  elif b == c and (d == b and e == b):
    return 'r'
  elif b != c:
    return 's'

def f_func(a1, a2, a3, a4):
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
        
        a1 = h_func(x0, x1, x6, x2)
        a2 = h_func(x0, x2, x7, x3)
        a3 = h_func(x0, x3, x8, x4)
        a4 = h_func(x0, x4, x5, x1)
        ret_img[row, col] = f_func(a1, a2, a3, a4)
  return ret_img 

def main():
  
  # usage: python ./hw6.py

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

  # Calculate Yokoi Connectivity Number for 4-connectivity
  img_Yokoi = Yokoi(img_down, neighbors)
  with open("Yokoi.txt", "w") as txt_file:
    for row in xrange(img_Yokoi.shape[0]):
      for col in xrange(img_Yokoi.shape[1]):
        if (img_Yokoi[row, col] < 0):
          txt_file.write(" ")
        else:
          txt_file.write("%r" % img_Yokoi[row, col])
      txt_file.write('\n')

if __name__ == '__main__':
  main()
