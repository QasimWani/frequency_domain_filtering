import cv2
from copy import deepcopy
import numpy as np
from math import pi,sin
from cmath import exp
import matplotlib.pyplot as plt

def gaussNoise(image):
      mean, sigma = 0, 0.1
      return image + np.random.normal(mean,sigma,image.shape)

plt.figure(figsize=(5*5, 2*5))
original_Image = cv2.imread("DIP.png", 0) #original_Image = cv2.imread("statue.jpg", 0)
fft_Image = np.fft.fftshift(np.fft.fft2(original_Image))
temp_img_c2, height, width = deepcopy(fft_Image), len(fft_Image), len(fft_Image[1])

T, a, b = 1, 0.1, 0.1
H = [[T if (var := pi * (u * a + v * b)) == 0 else (T * sin(var)/var) * exp(-1j * var) for v in range(width)] for u in range(height)]
temp_img_c2 = np.multiply(temp_img_c2, H)
output_Image = np.fft.ifft2(np.fft.ifftshift(temp_img_c2))
output_Image = abs(output_Image)
noisy_Image = gaussNoise(output_Image)

plt.subplot(151), plt.imshow(original_Image, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(np.log(1 + np.abs(fft_Image)), "gray"), plt.title("Centered FFT")
plt.subplot(153), plt.imshow(np.log(1 + np.abs(temp_img_c2)), "gray"), plt.title("Output Centered FFT")
plt.subplot(154), plt.imshow(np.abs(output_Image), "gray"), plt.title("Output Image")
plt.subplot(155), plt.imshow(np.abs(noisy_Image), "gray"), plt.title("Noisy Image")
plt.show()


