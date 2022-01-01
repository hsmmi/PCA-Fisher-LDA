import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


class Dataset():
    def __init__(self):
        self.image_size = None
        self.sample = None
        self.label = None
        self.label_name = None
        self.number_of_feature = None
        self.number_of_sample = None

    def read_dataset(self, folder: str, img_size: int,
                     visualize: bool = False):

        images = []
        name_feeling = []

        directory = os.path.abspath('')
        folder_path = os.path.join(directory, folder)
        for img_name in os.listdir(folder_path):
            name_feeling.append(img_name.split('.')[0:-1])
            img_path = os.path.join(folder_path, img_name)
            img_mat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_mat is not None:
                resized_img_mat = cv2.resize(img_mat, (img_size, img_size))
                images.append(resized_img_mat)

        images = np.array(images)
        name_feeling = np.array(name_feeling).reshape((-1, 3))
        number_of_image = name_feeling.shape[0]

        if visualize is True:
            for rnd in np.random.randint(number_of_image, size=5):
                plt.imshow(images[rnd],  cmap="gray")
                plt.title((
                    f'name: {name_feeling[rnd][0]}, ',
                    f'feeling: {name_feeling[rnd][1]}'))
                plt.show()

        vector_images = images.reshape(images.shape[0], -1)

        self.image_size = img_size
        self.sample = vector_images
        self.label = name_feeling
        self.label_name = np.array(['name', 'feeling'])
        self.number_of_feature = vector_images.shape[1]
        self.number_of_sample = vector_images.shape[0]
