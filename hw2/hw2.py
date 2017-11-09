import numpy as np
import cv2
import matplotlib.pyplot as plt

threshold = 128
area_threshold = 500

def binarize(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      if img[row][col] >= threshold:
        ret_img[row][col] = 255
      else:
        ret_img[row][col] = 0  
  return ret_img
  
def histogram(img):
  hist = np.zeros(256, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      hist[img[row, col]] += 1
  return hist

def drawhist(hist):
  fig = plt.figure()
  plt.plot(hist)
  plt.xlabel('Pixel value (0~255)')
  plt.ylabel('Number')
  plt.title('Histogram of lena.bmp')
  fig.savefig('histogram.png')
  #plt.show()
  return

def check_boundary(row, col, height, width):
  if row < 0 or row > height-1:
    return False
  elif col < 0 or col > width-1:
    return False
  return True

def iterative_cc(img):
  label = np.zeros(img.shape, int)
  label_init = 1
  changed = True

  # initialize each pixel with an unique label
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      if img[row, col] > 0: 
        label[row, col] = label_init
        label_init+=1

  while True:
    changed = False
    labelrows = label.shape[0]
    labelcols = label.shape[1]

    # the iteration of top-down pass
    for row in xrange(labelrows):
      for col in xrange(labelcols):
        # compare with its neighbors
        if label[row, col] > 0:
          min_label = label[row, col]
          # up neighbor
          if check_boundary(row-1, col, labelrows, labelcols):
            if label[row-1, col] > 0 and label[row-1, col] < min_label: 
              min_label = label[row-1, col]
          # left neighbor
          if check_boundary(row, col-1, labelrows, labelcols):
            if label[row, col-1] > 0 and label[row, col-1] < min_label:
              min_label = label[row, col-1]
          # down neighbor
          if check_boundary(row+1, col, labelrows, labelcols):
            if label[row+1, col] > 0 and label[row+1, col] < min_label:
              min_label = label[row+1, col]
          # right neighbor
          if check_boundary(row, col+1, labelrows, labelcols):
            if label[row, col+1] > 0 and label[row, col+1] < min_label:
              min_label = label[row, col+1]

          if min_label < label[row, col]:
            label[row, col] = min_label
            changed = True          

    # the iteration of bottom-up pass
    for row in range(labelrows-1, 0, -1):
      for col in range(labelcols-1, 0, -1):
        # compare with its neighbors
        if label[row, col] > 0:
          min_label = label[row, col]
          # up neighbor
          if check_boundary(row-1, col, labelrows, labelcols):
            if label[row-1, col] > 0 and label[row-1, col] < min_label:
              min_label = label[row-1, col]
              changed = True
          # left neighbor
          if check_boundary(row, col-1, labelrows, labelcols):
            if label[row, col-1] > 0 and label[row, col-1] < min_label:
              min_label = label[row, col-1]
              changed = True
          # down neighbor
          if check_boundary(row+1, col, labelrows, labelcols):
            if label[row+1, col] > 0 and label[row+1, col] < min_label:
              min_label = label[row+1, col]
              changed = True
          # right neighbor
          if check_boundary(row, col+1, labelrows, labelcols):
            if label[row, col+1] > 0 and label[row, col+1] < min_label:
              min_label = label[row, col+1]
              changed = True
          
          if min_label < label[row, col]:
            label[row, col] = min_label
            changed = True          

    if changed == False:
      break    

  # For the connected components, record the label count of each component
  label_count = np.zeros(np.max(label)+1, int)  
  for row in xrange(labelrows):
    for col in xrange(labelcols):
      if label[row, col] > 0:
        label_count[label[row, col]] += 1
  
  # Get the bounding box for each componet 
  # and omit components with the number of the label count less than threshold(500)     
  for i in xrange(1, label_count.shape[0]):
    if label_count[i] >= 500:
      # i is the region id 
      id_set = np.array(np.where(label == i)).T
      # locate the up-left point and the bottom-right point
      id_up_left = np.array([np.min(id_set[:, 0]), np.min(id_set[:, 1])])
      id_bottom_right = np.array([np.max(id_set[:, 0]), np.max(id_set[:, 1])])
      # draw 4 lines with pixel value = 128
      img[id_up_left[0]:id_bottom_right[0] + 1, id_up_left[1]] = 128
      img[id_up_left[0]:id_bottom_right[0] + 1, id_bottom_right[1]] = 128
      img[id_up_left[0], id_up_left[1]:id_bottom_right[1] + 1] = 128
      img[id_bottom_right[0], id_up_left[1]:id_bottom_right[1] + 1] = 128   

  return img

def main():
  
  # usage: python ./hw2.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # Do binarize
  img_bin = binarize(img)
  cv2.imwrite('binary.bmp', img_bin)
  
  # hist is an array that records the distribution of pixel values (0~255) 
  hist = histogram(img)
  
  # Draw histogram
  drawhist(hist)

  # Find connected components from binary image
  # use iterative algorithm and four-connected neighbor
  print 'Find 4-connected connected components. Please wait a moment.'
  img_cc = iterative_cc(img_bin)
  cv2.imwrite('connected4.bmp', img_cc)

if __name__ == '__main__':
  main()
