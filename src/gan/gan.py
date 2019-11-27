from __future__ import print_function

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import sys

# TODO find a better way to import
# path needs to be changed depending on every system
sys.path.append('/home/steffen/PycharmProjects/PycharmProjects/neuronart/src/scripts/dataProcessing')
from dataProcessing import CifarImage, LMDBase

np.random.seed(1337)


def build_generator(random_size, number_classes):
    # create sequential model
    cnn = Sequential()
    cnn.add(Dense(3 * 3 * 384, input_dim=random_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))
    cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # create input class
    latent = Input(shape=(random_size,))
    image_class = Input(shape=(1,), dtype='int32')

    cls = Embedding(number_classes, random_size,
                    embeddings_initializer='glorot_normal')(image_class)

    h = layers.multiply([latent, cls])
    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator(number_classes):
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))
    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(number_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])


def get_train_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    number_train, number_test = x_train.shape[0], x_test.shape[0]

    return x_train, y_train, number_train, x_test, y_test, number_test


def get_train_data_images(train_size, lmdb_size, lmdb_path):
    """
    Returns all train data from database.

    :param train_size: The number of train images
    :param lmdb_size: The size the database in bytes
    :param lmdb_path: The path to the database
    :return: An array of images
    """
    # open lmdb database
    lmdb = LMDBase(lmdb_path, lmdb_size)

    # read train data
    images, labels = lmdb.read_many_lmdb(image_id_start=1,
                                         image_id_end=train_size)

    # close lmdb database
    lmdb.close_lmdb()

    # return images
    return images


def get_test_data_images(train_size, max_images, lmdb_size, lmdb_path):
    """
    Returns all test data from database.

    :param train_size: The number of train images
    :param max_images: The number of images in the database
    :param lmdb_size: The size the database in bytes
    :param lmdb_path: The path to the database
    :return: An array of images
    """
    # open lmdb database
    lmdb = LMDBase(lmdb_path, lmdb_size)

    # read train data
    images, labels = lmdb.read_many_lmdb(image_id_start=train_size+1,
                                         image_id_end=max_images)

    # close lmdb database
    lmdb.close_lmdb()

    # return images
    return images


def get_train_data_batch_images(batch_number, batch_size, lmdb_size, lmdb_path):
    """
    Reads a batch from the image database.

    :param batch_number: The index of the needed batch
    :param batch_size: The number of images in a batch
    :param lmdb_size: The size the database in bytes
    :param lmdb_path: The path to the database
    :return: An array of images
    """
    # open lmdb database
    lmdb = LMDBase(lmdb_path, lmdb_size)

    # read actual batch
    images, labels = lmdb.read_many_lmdb(image_id_start=batch_number * batch_size - batch_size + 1,
                                         image_id_end=batch_number * batch_size)

    # close lmdb database
    lmdb.close_lmdb()

    # return images
    return images


if __name__ == '__main__':
    epochs = 100
    batch_size = 100
    test_size = 1100
    train_size = 6100
    random_size = 100
    number_classes = 2
    number_rows_example_image = 40
    example_images = False
    lmdb_path = '/home/steffen/PycharmProjects/PycharmProjects/neuronart/src/scripts/dataProcessing/data/lmdb'
    lmdb_size = 200000 * 7500 * 10

    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator(number_classes)
    discriminator.compile(
        optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(random_size, number_classes)

    # create combined model
    random_input = Input(shape=(random_size,))
    image_class = Input(shape=(1,), dtype='int32')
    generator_combined = generator([random_input, image_class])
    discriminator.trainable = False
    fake, aux = discriminator(generator_combined)
    combined = Model([random_input, image_class], [fake, aux])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get the train data
    # x_train, y_train, number_train, x_test, y_test, number_test = get_train_data()

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print("Epoch " + str(epoch) + " of " + str(epochs))

        # get number of batches
        num_batches = int(np.ceil(train_size / float(batch_size)))
        print("Number batches train: " + str(num_batches))

        # setup progress bar
        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # get actual batch images and labels
            image_batch = get_train_data_batch_images(index + 1, batch_size, lmdb_size, lmdb_path)
            label_batch = np.repeat(1, batch_size)

            # create some noise for the generator
            noise = np.random.uniform(-1, 1, (len(image_batch), random_size))
            sampled_labels = np.random.randint(0, number_classes, len(image_batch))

            # generate images
            generated_images = generator.predict(
                [noise, sampled_labels.reshape(-1, 1)], verbose=0)

            # create some input for discriminator out of generated and real images
            x = np.concatenate((image_batch, generated_images))

            # create labels for generated and real images
            soft_zero, soft_one = 0, 0.95
            y = np.array(
                [soft_one] * len(image_batch) + [soft_zero] * len(image_batch))
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # create sample weights for discriminator
            disc_sample_weight = [np.ones(2 * len(image_batch)),
                                  np.concatenate((np.ones(len(image_batch)) * 2,
                                                  np.zeros(len(image_batch))))]

            # train discriminator on generated and real images from actual batch. Save discriminator loss.
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # create some noise and labels for combined model
            noise = np.random.uniform(-1, 1, (2 * len(image_batch), random_size))
            sampled_labels = np.random.randint(0, number_classes, 2 * len(image_batch))
            trick = np.ones(2 * len(image_batch)) * soft_one

            # train combined model on generated and real images from actual batch. Save gen loss.
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

            # update progress bar
            progress_bar.update(index + 1)

        print("Testing phase of epoch " + str(epoch) + ":")

        # create some noise
        noise = np.random.uniform(-1, 1, (test_size, random_size))
        sampled_labels = np.random.randint(0, number_classes, test_size)

        # generate images
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        # get test data and test labels
        test_images = get_test_data_images(6100, 7200, lmdb_size, lmdb_path)
        test_labels = np.repeat(1, len(test_images))

        # create image batch from test images and generated ones
        x = np.concatenate((test_images, generated_images))
        y = np.array([1] * test_size + [0] * test_size)
        aux_y = np.concatenate((test_labels, sampled_labels), axis=0)

        # check if discriminator can recognise generated images
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        # save the loss for discriminator
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # generate images
        noise = np.random.uniform(-1, 1, (2 * test_size, random_size))
        sampled_labels = np.random.randint(0, number_classes, 2 * test_size)
        trick = np.ones(2 * test_size)

        # evaluate the combined model
        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        # save the loss of generator
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # save the results
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        # print actual results
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save the weights of the actual epoch
        generator.save_weights(
            'weights/generator/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'weights/discriminator/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # save some generated images to check results while training
        if example_images:
            # create noise and labels
            noise = np.tile(np.random.uniform(-1, 1, (number_rows_example_image, random_size)),
                            (number_classes, 1))
            sampled_labels = np.array([
                [i] * number_rows_example_image for i in range(number_classes)
            ]).reshape(-1, 1)

            # generate some images
            generated_images = generator.predict(
                [noise, sampled_labels], verbose=0)

            # get some real images
            real_labels = y_train[(epoch - 1) * number_rows_example_image * number_classes:
                                  epoch * number_rows_example_image * number_classes]
            indices = np.argsort(real_labels, axis=0)
            real_images = x_train[(epoch - 1) * number_rows_example_image * number_classes:
                                  epoch * number_rows_example_image * number_classes][indices]

            # create an image containing the generated and real images
            img = np.concatenate(
                (generated_images,
                 np.repeat(np.ones_like(x_train[:1]), number_rows_example_image, axis=0),
                 real_images))
            img = (np.concatenate([r.reshape(-1, 28)
                                   for r in np.split(img, 2 * number_classes + 1)
                                   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

            # save the results
            Image.fromarray(img).save(
                'images/plot_epoch_{0:03d}_generated.png'.format(epoch))

    # save history after training finished
    with open('acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)
