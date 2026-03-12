import cv2
import os
import numpy as np

def load_images(data_dir):

    images = []
    labels = []

    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            img = cv2.imread(path)
            img = cv2.resize(img,(64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.flatten()

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)
