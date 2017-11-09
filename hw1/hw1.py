import sys
import numpy as np
import cv2

def upside_down(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    ret_img[row, :] = img[img.shape[0]-row-1, :]  
  return ret_img
  
def right_side_left(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      ret_img[row, col] = img[row, img.shape[1]-col-1]
  return ret_img

def dia_mirror(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      ret_img[row, col] = img[col, row]
  return ret_img

def main():
  
  # usage: python ./hw1.py [task_id]
  # task_id = 1~3 
  assert (len(sys.argv) == 2), "Please input task_id(1~3)."
  assert (int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 3), "Task_id should be between 1~3."

  # img is a 512*512 array
  img = cv2.imread('lena.bmp')

  if sys.argv[1] == '1':
    # Do upside-down
    img_res = upside_down(img)
    cv2.imwrite('upside_down.bmp', img_res)
  
  elif sys.argv[1] == '2':
    # Do right-side-left
    img_res = right_side_left(img)
    cv2.imwrite('right_side_left.bmp', img_res)
 
  else:
    # Do diagnally mirror
    img_res = dia_mirror(img)
    cv2.imwrite('dia_mirror.bmp', img_res)
 
if __name__ == '__main__':
  main()
