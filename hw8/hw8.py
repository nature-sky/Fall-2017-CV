import numpy as np
import cv2, math, random

def gaussian_noise(img, amplitude):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      ret_img[row, col] = img[row, col] + amplitude * random.gauss(0, 1)
  return ret_img
  
def salt_and_pepper_noise(img, threshold):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]): 
    for col in xrange(img.shape[1]):
      if (random.uniform(0, 1) < threshold):
        ret_img[row, col] = 0
      elif (random.uniform(0, 1) > (1-threshold)):
        ret_img[row, col] = 255
      else:
        ret_img[row, col] = img[row, col]
  return ret_img

def snr(img, img_noise):
  u = np.mean(img)
  un = np.mean(img_noise - img)
  vs = 0
  vn = 0

  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      vs += math.pow(img[row, col] - u, 2)
  vs /= (img.shape[0] * img.shape[1])
   
  for row in xrange(img_noise.shape[0]):
    for col in xrange(img_noise.shape[1]):
      vn += math.pow(img_noise[row, col] - img[row, col] - un, 2)
  vn /= (img_noise.shape[0] * img_noise.shape[1])
  snr = 20 * math.log10(math.sqrt(vs) / math.sqrt(vn))
  return snr

def box_filter(img, fsize):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      ret_img[row, col] = np.mean(img[row: row + fsize, col: col+fsize])
  return ret_img

def median_filter(img, fsize):
  ret_img = np.zeros(img.shape, int)
  for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
      ret_img[row, col] = np.median(img[row: row + fsize, col: col+fsize])
  return ret_img

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
  
  # usage: python ./hw8.py

  # img is a 512*512 array
  img = cv2.imread('lena.bmp', 0)

  # Do Gaussian noise with amplitude 10 and 30
  img_gau10 = gaussian_noise(img, 10)
  cv2.imwrite('gau_noise10.bmp', img_gau10)  
  img_gau30 = gaussian_noise(img, 30)
  cv2.imwrite('gau_noise30.bmp', img_gau30)
  print 'Task 1 finished. \r\n'
  
  # Do salt_and_pepper noise with threshold 0.05 and 0.1
  img_salt005 = salt_and_pepper_noise(img, 0.05)
  cv2.imwrite('salt_noise005.bmp', img_salt005)
  img_salt01 = salt_and_pepper_noise(img, 0.1)
  cv2.imwrite('salt_noise01.bmp', img_salt01)
  print 'Task 2 finished. \r\n'
  
  # Do box filter with 3*3 and 5*5 block size.
  #   With 3*3 block size
  img_box3x3_gau10 = box_filter(img_gau10, 3)
  cv2.imwrite('box3x3_gau10.bmp', img_box3x3_gau10)
  img_box3x3_gau30 = box_filter(img_gau30, 3)
  cv2.imwrite('box3x3_gau30.bmp', img_box3x3_gau30)
  img_box3x3_salt005 = box_filter(img_salt005, 3)
  cv2.imwrite('box3x3_salt005.bmp', img_box3x3_salt005)
  img_box3x3_salt01 = box_filter(img_salt01, 3)
  cv2.imwrite('box3x3_salt01.bmp', img_box3x3_salt01)
  #   With 5*5 block size
  img_box5x5_gau10 = box_filter(img_gau10, 5)
  cv2.imwrite('box5x5_gau10.bmp', img_box5x5_gau10)
  img_box5x5_gau30 = box_filter(img_gau30, 5)
  cv2.imwrite('box5x5_gau30.bmp', img_box5x5_gau30)
  img_box5x5_salt005 = box_filter(img_salt005, 5)
  cv2.imwrite('box5x5_salt005.bmp', img_box5x5_salt005)
  img_box5x5_salt01 = box_filter(img_salt01, 5)
  cv2.imwrite('box5x5_salt01.bmp', img_box5x5_salt01)
  print 'Task 3 finished. \r\n'

  # Do median filter with 3*3 and 5*5 block size.
  #   With 3*3 block size
  img_median3x3_gau10 = median_filter(img_gau10, 3)
  cv2.imwrite('median3x3_gau10.bmp', img_median3x3_gau10)
  img_median3x3_gau30 = median_filter(img_gau30, 3)
  cv2.imwrite('median3x3_gau30.bmp', img_median3x3_gau30)
  img_median3x3_salt005 = median_filter(img_salt005, 3)
  cv2.imwrite('median3x3_salt005.bmp', img_median3x3_salt005)
  img_median3x3_salt01 = median_filter(img_salt01, 3)
  cv2.imwrite('median3x3_salt01.bmp', img_median3x3_salt01)
  #   With 5*5 block size
  img_median5x5_gau10 = median_filter(img_gau10, 5)
  cv2.imwrite('median5x5_gau10.bmp', img_median5x5_gau10)
  img_median5x5_gau30 = median_filter(img_gau30, 5)
  cv2.imwrite('median5x5_gau30.bmp', img_median5x5_gau30)
  img_median5x5_salt005 = median_filter(img_salt005, 5)
  cv2.imwrite('median5x5_salt005.bmp', img_median5x5_salt005)
  img_median5x5_salt01 = median_filter(img_salt01, 5)
  cv2.imwrite('median5x5_salt01.bmp', img_median5x5_salt01)
  print 'Task 4 finished. \r\n'

  # kernel is a 3-5-5-5-3 octagon
  # the origin is at the center
  kernel = [[-2, -1], [-2, 0], [-2, 1],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -1], [2, 0], [2, 1]]

  # Do opening then closing
  img_opening_then_closing_gau10 = closing(opening(img_gau10, kernel), kernel)
  cv2.imwrite('opening_then_closing_gau10.bmp', img_opening_then_closing_gau10)
  img_opening_then_closing_gau30 = closing(opening(img_gau30, kernel), kernel)
  cv2.imwrite('opening_then_closing_gau30.bmp', img_opening_then_closing_gau30)
  img_opening_then_closing_salt005 = closing(opening(img_salt005, kernel), kernel)
  cv2.imwrite('opening_then_closing_salt005.bmp', img_opening_then_closing_salt005)
  img_opening_then_closing_salt01 = closing(opening(img_salt01, kernel), kernel)
  cv2.imwrite('opening_then_closing_salt01.bmp', img_opening_then_closing_salt01)
  
  # Do closing then opening
  img_closing_then_opening_gau10 = opening(closing(img_gau10, kernel), kernel)
  cv2.imwrite('closing_then_opening_gau10.bmp', img_closing_then_opening_gau10)
  img_closing_then_opening_gau30 = opening(closing(img_gau30, kernel), kernel)
  cv2.imwrite('closing_then_opening_gau30.bmp', img_closing_then_opening_gau30)
  img_closing_then_opening_salt005 = opening(closing(img_salt005, kernel), kernel)
  cv2.imwrite('closing_then_opening_salt005.bmp', img_closing_then_opening_salt005)
  img_closing_then_opening_salt01 = opening(closing(img_salt01, kernel), kernel)
  cv2.imwrite('closing_then_opening_salt01.bmp', img_closing_then_opening_salt01)
  print 'Task 5 finished. \r\n'

  # Calculate SNR for each instance.
  with open("snr.txt", "w") as txt_file:
    txt_file.write("gau_noise10: " + str(snr(img, img_gau10)) + '\n')
    txt_file.write("gau_noise30: " + str(snr(img, img_gau30)) + '\n')
    txt_file.write("salt_noise005: " + str(snr(img, img_salt005)) + '\n')
    txt_file.write("salt_noise01: " + str(snr(img, img_salt01)) + '\n')

    txt_file.write("box3x3_gau10: " + str(snr(img, img_box3x3_gau10)) + '\n')
    txt_file.write("box3x3_gau30: " + str(snr(img, img_box3x3_gau30)) + '\n')
    txt_file.write("box3x3_salt005: " + str(snr(img, img_box3x3_salt005)) + '\n')
    txt_file.write("box3x3_salt01: " + str(snr(img, img_box3x3_salt01)) + '\n')

    txt_file.write("box5x5_gau10: " + str(snr(img, img_box5x5_gau10)) + '\n')
    txt_file.write("box5x5_gau30: " + str(snr(img, img_box5x5_gau30)) + '\n')
    txt_file.write("box5x5_salt005: " + str(snr(img, img_box5x5_salt005)) + '\n')
    txt_file.write("box5x5_salt01: " + str(snr(img, img_box5x5_salt01)) + '\n')

    txt_file.write("median3x3_gau10: " + str(snr(img, img_median3x3_gau10)) + '\n')
    txt_file.write("median3x3_gau30: " + str(snr(img, img_median3x3_gau30)) + '\n')
    txt_file.write("median3x3_salt005: " + str(snr(img, img_median3x3_salt005)) + '\n')
    txt_file.write("median3x3_salt01: " + str(snr(img, img_median3x3_salt01)) + '\n')

    txt_file.write("median5x5_gau10: " + str(snr(img, img_median5x5_gau10)) + '\n')
    txt_file.write("median5x5_gau30: " + str(snr(img, img_median5x5_gau30)) + '\n')
    txt_file.write("median5x5_salt005: " + str(snr(img, img_median5x5_salt005)) + '\n')
    txt_file.write("median5x5_salt01: " + str(snr(img, img_median5x5_salt01)) + '\n')

    txt_file.write("opening_then_closing_gau10: " + str(snr(img, img_opening_then_closing_gau10)) + '\n')
    txt_file.write("opening_then_closing_gau30: " + str(snr(img, img_opening_then_closing_gau30)) + '\n')
    txt_file.write("opening_then_closing_salt005: " + str(snr(img, img_opening_then_closing_salt005)) + '\n')
    txt_file.write("opening_then_closing_salt01: " + str(snr(img, img_opening_then_closing_salt01)) + '\n')

    txt_file.write("closing_then_opening_gau10: " + str(snr(img, img_closing_then_opening_gau10)) + '\n')
    txt_file.write("closing_then_opening_gau30: " + str(snr(img, img_closing_then_opening_gau30)) + '\n')
    txt_file.write("closing_then_opening_salt005: " + str(snr(img, img_closing_then_opening_salt005)) + '\n')
    txt_file.write("closing_then_opening_salt01: " + str(snr(img, img_closing_then_opening_salt01)) + '\n')
  print 'SNR finished. \r\n'

if __name__ == '__main__':
  main()
