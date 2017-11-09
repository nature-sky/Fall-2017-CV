import numpy as np
import cv2
import matplotlib.pyplot as plt

threshold = 128
area_threshold = 500

def divide_3(img):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
        ret_img[row][col] = img[row, col] / 3
  return ret_img
  
def histogram(img):
  hist = np.zeros(256, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      hist[img[row, col]] += 1
  return hist

def drawhist(hist, src, dst):
  fig = plt.figure()
  plt.plot(hist)
  plt.xlabel('Pixel value (0~255)')
  plt.ylabel('Number')
  plt.title('Histogram of '+ src)
  fig.savefig(dst)
  #plt.show()
  return

def hist_equal (img, hist):
  ret_img = np.zeros(img.shape, int)
  
  # compute the number of intensity n_j
  hist_intensity = np.zeros(256, int)
  hist_intensity[0] = hist[0]
  for i in range(1, 256):
    hist_intensity[i] = hist_intensity[i-1] + hist[i]

  # compute s_k =
  # 255 * (the accumulated number of pixel value / the total number of image)
  hist_intensity = 255 * hist_intensity / np.sum(hist)
  
  # Replace each pixel with s_k
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):    
      ret_img[row, col] =  hist_intensity[img[row, col]]
    
  return ret_img

def main():
  
  # usage: python ./hw3.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # divide all pixel value of the original image by 3
  img_divided_3 = divide_3(img)
  cv2.imwrite('divided_3.bmp', img_divided_3)
  
  # Records the histogram before equalization 
  hist_before = histogram(img_divided_3)
  
  # Draw histogram before equalized
  drawhist(hist_before, 'divided_3.bmp', 'histogram_before_equalized.png')

  # Do histogram equalization for 'divided_3.bmp'
  img_after_equalized = hist_equal(img_divided_3, hist_before)

  # Output image after equalization
  cv2.imwrite('after_equalized.bmp', img_after_equalized)
  
  # Records the histogram after equalization 
  hist_after = histogram(img_after_equalized)
  
  # Draw histogram after equalized
  drawhist(hist_after, 'after_equalized.bmp', 'histogram_after_equalized.png')

if __name__ == '__main__':
  main()
