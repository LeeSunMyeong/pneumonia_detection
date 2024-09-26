# utils/utils.py
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

def visualize_sample_images(directory, num_images=5):
    images = random.sample(os.listdir(directory), num_images)
    for img in images:
        img_path = os.path.join(directory, img)
        image = Image.open(img_path).convert('RGB')
        plt.imshow(image)
        plt.axis('off')
        plt.show()
