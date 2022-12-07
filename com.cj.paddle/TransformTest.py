import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import CenterCrop

transform = CenterCrop(224)

image = cv2.imread('flower_demo.jpg')

image_after_transform = transform(image)
plt.subplot(1, 2, 1)
plt.title('origin image')
plt.imshow(image[:, :, ::-1])
plt.subplot(1, 2, 2)
plt.title('CenterCrop image')
plt.imshow(image_after_transform[:, :, ::-1])
plt.show()