# ! wget https://paddle-imagenet-models-name.bj.bcebos.com/data/demo_images/flower_demo.png


import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import CenterCrop

transform = CenterCrop(224)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('CenterCrop image')
plt.imshow(image_after_transform[:,:,::-1])




import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import RandomHorizontalFlip

transform = RandomHorizontalFlip(0.5)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('RandomHorizontalFlip image')
plt.imshow(image_after_transform[:,:,::-1])



import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle.vision.transforms import ColorJitter

transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

image = cv2.imread('flower_demo.png')

image_after_transform = transform(image)
plt.subplot(1,2,1)
plt.title('origin image')
plt.imshow(image[:,:,::-1])
plt.subplot(1,2,2)
plt.title('ColorJitter image')
plt.imshow(image_after_transform[:,:,::-1])
