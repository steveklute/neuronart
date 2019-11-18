import numpy as np
import lmdb
import pickle


class CifarImage:
    """
    A class for saving images in a LMDB
    """
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


class LMDBase:
    """

    """

    def __init__(self, folder):
        self.folder = folder

    def store_single_lmdb(image, image_id, label):
        """
        Stores a single image to a LMDB.

        Parameters:
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
        """
        map_size = image.nbytes * 10

        # Create a new LMDB environment
        env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

        # Start a new write transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be strings
            value = CifarImage(image, label)
            key = f"{image_id:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
        env.close()

    def store_many_lmdb(images, labels):
        """
        Stores an array of images to LMDB.
        Parameters:

        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
        """
        num_images = len(images)

        map_size = num_images * images[0].nbytes * 10

        # Create a new LMDB DB for all the images
        env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

        # Same as before — but let's write all the images in a single transaction
        with env.begin(write=True) as txn:
            for i in range(num_images):
                # All key-value pairs need to be Strings
                value = CifarImage(images[i], labels[i])
                key = f"{i:08}"
                txn.put(key.encode("ascii"), pickle.dumps(value))
        env.close()

    def read_single_lmdb(image_id):
        """
        Stores a single image to LMDB.

        Parameters:
        image_id    integer unique ID for image

        Returns:
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
        """
        # Open the LMDB environment
        env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

        # Start a new read transaction
        with env.begin() as txn:
            # Encode the key the same way as we stored it
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember it's a CIFAR_Image object that is loaded
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            image = cifar_image.get_image()
            label = cifar_image.label
        env.close()

        return image, label


    def read_many_lmdb(num_images):
        """
        Reads image from LMDB.

        Parameters:

        num_images   number of images to read

        Returns:
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
        """
        images, labels = [], []
        env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

        # Start a new read transaction
        with env.begin() as txn:
            # Read all images in one single transaction, with one lock
            # We could split this up into multiple transactions if needed
            for image_id in range(num_images):
                data = txn.get(f"{image_id:08}".encode("ascii"))
                # Remember that it's a CIFAR_Image object
                # that is stored as the value
                cifar_image = pickle.loads(data)
                # Retrieve the relevant bits
                images.append(cifar_image.get_image())
                labels.append(cifar_image.label)
        env.close()
        return images, labels
