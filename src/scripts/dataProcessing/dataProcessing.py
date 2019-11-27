import numpy as np
import lmdb
import pickle
import os
import os.path
import math
from matplotlib.image import imread
import cv2
from keras.utils.generic_utils import Progbar


class CifarImage:
    """
    A class for saving images in a LMDB
    """

    def __init__(self, image, label):
        """
        Creates an image for lmdb.

        :param image: The image
        :param label: The label of the image
        """
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """
        Returns the image as a numpy array.

        :return: the image as a numpy array
        """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


class LMDBase:
    """
    The main class for LMDB.
    """

    def __init__(self, folder, mapsize_bytes, read_only=False):
        self.folder = folder
        self.max_id = 0
        self.mapsize_bytes = mapsize_bytes
        self.read_only = read_only

        # create lmdb environment
        if read_only:
            self.env = lmdb.open(folder, readonly=True)
        else:
            self.env = lmdb.open(folder, map_size=mapsize_bytes)

    def close_lmdb(self):
        """
        Closes the LMDB instance.

        :return: nothing
        """
        self.env.close()

    def store_single_lmdb(self, image, label):
        """
        Stores a single image ti LMDB.

        :param image: The image.
        :param label: The label of the image.
        :return: nothing
        """

        # create new instance if in read only mode
        if self.read_only:
            self.mapsize_bytes = image.nbytes * 10
            self.env = lmdb.open(self.folder, map_size=self.mapsize_bytes)

        # open a new write transaction and save the image
        with self.env.begin(write=True) as txn:
            # create image object
            value = CifarImage(image, label)

            # create a key for the image
            self.max_id += 1
            key = str(self.max_id).rjust(8, "0")

            # save the image
            txn.put(key.encode("ascii"), pickle.dumps(value))

    def store_many_lmdb(self, images, labels):
        """
        Stores multiple images in LMDB.

        :param images: An array of images
        :param labels: An array of labels.
        :return: nothing
        """
        num_images = len(images)

        # create new instance if in read only mode
        if self.read_only:
            self.mapsize_bytes = num_images * self.mapsize_bytes * 10
            self.env = lmdb.open(self.folder, map_size=self.mapsize_bytes)

        # open a new write transaction and save the image
        with self.env.begin(write=True) as txn:
            for i in range(num_images):
                # create an image object
                value = CifarImage(images[i], labels[i])

                # create key for the image
                self.max_id += 1
                key = str(self.max_id).rjust(8, "0")

                # save the image
                txn.put(key.encode("ascii"), pickle.dumps(value))

    def read_single_lmdb(self, image_id):
        """
        Reads a single image by a given id.

        :param image_id: The id of the image. The ids starts by 1.
        :return: a tupel of (image, label)
        """

        # open a new read transaction and read the image
        with self.env.begin() as txn:
            # create a key and read the image
            data = txn.get(str(image_id).rjust(8, "0").encode("ascii"))

            # convert to object as which the image got saved
            cifar_image = pickle.loads(data)

            # load the relevant data
            image = cifar_image.get_image()
            label = cifar_image.label

        return image, label

    def read_many_lmdb(self, image_id_start, image_id_end):
        """
        Reads images from LMDB.

        :param image_id_start: The first id of image sequence. Image ids start by 1 not 0.
        :param image_id_end: The last id of image sequence.
        :return: a tupel of (images array, labels array)
        """
        images, labels = [], []

        # open a new read transaction and read the images
        with self.env.begin() as txn:
            for image_id in range(image_id_start, image_id_end+1):
                # create a key and read the image
                data = txn.get(str(image_id).rjust(8, "0").encode("ascii"))

                # convert to object as which the image got saved
                cifar_image = pickle.loads(data)

                # load the relevant data and append it to arrays
                images.append(cifar_image.get_image())
                labels.append(cifar_image.label)

        return images, labels

    def store_images_from_folder(self, path, height, width, batch_size):
        """
        This method reads all images from a given directory , resize and save them.

        :param path: The path with the images.
        :param height The new height of all images
        :param width The new width of all images
        :param batch_size: The max number of all images in the ram, before they get saved.
                            Set it depending on image and ram size.
        :return: True if images are saved successfully
        """
        valid_images = [".jpg", ".gif", ".png", ".tga"]
        index = 0
        images = []
        labels = []

        # create progress bar to visualize progress
        progress_bar = Progbar(target=len(os.listdir(path)))
        print("Progressing " + str(len(os.listdir(path))) + " images")

        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]

            # check if image has valid data type
            if ext.lower() not in valid_images:
                continue

            # load image
            actual_image = imread(os.path.join(path, f))

            # resize images
            resized_image = self.resize_img(actual_image, height, width)

            # save images
            images.append(resized_image)
            labels.append(f)

            # update progressbar
            progress_bar.update(index + 1)
            index += 1

            # save batch
            if index % batch_size == 0:
                # save the batch
                self.store_many_lmdb(images, labels)
                # clear arrays
                images = []
                labels = []

        # store images from last batch
        if len(images) > 0:
            # save the batch
            self.store_many_lmdb(images, labels)

        print("All images are stored!")
        return True

    @staticmethod
    def resize_img_max_pixel(image, max_pixel):
        """
        Resizes the given image approximately to a given number of pixel, while the image keeps its ratio.

        :param image: The image which should be resized
        :param max_pixel: The overall number of pixels
        :return: The resized image array
        """
        h_src = image.shape[0]
        w_src = image.shape[1]
        h_new = 0
        w_new = 0

        if h_src > w_src:
            ratio = h_src / w_src
            h_new = int(math.sqrt(max_pixel * ratio))
            w_new = int(math.sqrt(max_pixel / ratio))
            if h_new*w_new < max_pixel:
                h_new += 1
        else:
            ratio = w_src / h_src
            h_new = int(math.sqrt(max_pixel / ratio))
            w_new = int(math.sqrt(max_pixel * ratio))
            if h_new*w_new < max_pixel:
                w_new += 1

        resized_image = cv2.resize(image, dsize=(h_new, w_new), interpolation=cv2.INTER_CUBIC)

        return resized_image

    @staticmethod
    def resize_img(image, height, width):
        """
        Resizes the image. All image content will be kept but original ratio will be lost.

        :param image: The image
        :param height: The new height for the image
        :param width: The new width for the image
        :return: The resized image
        """
        return cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    print("dataProcessing loaded!")
    # create lmdb instance
    # lmdb = LMDBase('data/lmdb', 200000 * 7500 * 10)

    # read images from folder
    # lmdb.store_images_from_folder('data/landscape_painting', 500, 500, 100)

    # TODO remove test code later
    # img0 = imread('data/landscape_painting/6.jpg')
    # img1 = imread('data/landscape_painting/7.jpg')
    # img2 = imread('data/landscape_painting/8.jpg')
    # resized_image0 = lmdb.resize_img(img0, 500, 500)
    # resized_image1 = lmdb.resize_img(img1, 500, 500)
    # resized_image2 = lmdb.resize_img(img2, 500, 500)
    # lmdb.store_single_lmdb(resized_image0, "img0")
    # lmdb.store_single_lmdb(resized_image1, "img1")
    # lmdb.store_single_lmdb(resized_image2, "img2")

    # read_img0, label0 = lmdb.read_single_lmdb("1")
    # read_img1, label1 = lmdb.read_single_lmdb("2")
    # read_img2, label2 = lmdb.read_single_lmdb("3")

    # Image.fromarray(read_img0).save('img0.jpg')
    # Image.fromarray(read_img1).save('img1.jpg')
    # Image.fromarray(read_img2).save('img2.jpg')
