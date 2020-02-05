import os
import sys

import keras
from keras import layers
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.utils import multi_gpu_model

import numpy as np

import json

# TODO find a better way to import
# path needs to be changed depending on every system
sys.path.append('/home/steffen/neuronart/scripts/dataProcessing')
from dataProcessing import CifarImage, LMDBase


def build_generator(channels, latent_dim):
    generator_input = keras.Input(shape=(latent_dim,))
    gen = layers.Dense(128 * 16 * 16)(generator_input)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Reshape((16, 16, 128))(gen)
    gen = layers.Conv2D(256, 5, padding='same')(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Conv2DTranspose(256, 4, strides=3, padding='same')(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Conv2DTranspose(256, 4, strides=3, padding='same')(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Conv2D(channels, 7, activation='tanh', padding='same')(gen)
    generator = keras.models.Model(generator_input, gen)
    # use multiple gpus if available
    try:
        generator = multi_gpu_model(generator, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    generator.summary()
    return generator


def build_generator_gen2(channels, latent_dim):
    generator_input = keras.Input(shape=(latent_dim,))
    gen = layers.Dense(4 * 4 * 1024, input_dim=latent_dim, activation='relu')(generator_input)
    gen = layers.ReLU()(gen)
    gen = layers.Reshape((4, 4, 1024))(gen)
    gen = layers.Conv2DTranspose(512, 5, padding='valid', kernel_initializer='glorot_normal')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.Conv2DTranspose(256, 5, strides=3, padding='same', kernel_initializer='glorot_normal')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.Conv2DTranspose(128, 5, strides=2, padding='same', kernel_initializer='glorot_normal')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.Conv2DTranspose(64, 5, strides=3, padding='same', kernel_initializer='glorot_normal')(gen)
    gen = layers.ReLU()(gen)
    gen = layers.Conv2DTranspose(channels, 7, strides=2, activation='tanh', padding='same', kernel_initializer='glorot_normal') \
        (gen)
    generator = keras.models.Model(generator_input, gen)
    # use multiple gpus if available
    try:
        generator = multi_gpu_model(generator, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    generator.summary()
    return generator


def build_discriminator(height, width, channels):
    discriminator_input = layers.Input(shape=(height, width, channels))
    dis = layers.Conv2D(128, 3)(discriminator_input)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.4)(dis)
    dis = layers.Dense(1, activation='sigmoid')(dis)
    discriminator = keras.models.Model(discriminator_input, dis)
    # use multiple gpus if available
    try:
        discriminator = multi_gpu_model(discriminator, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    discriminator.summary()
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008, clipvalue=1.0, decay=1e-8
    )
    discriminator.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy")
    discriminator.trainable = False
    return discriminator


def build_discriminator_gen2(height, width, channels):
    discriminator_input = layers.Input(shape=(height, width, channels))
    dis = layers.Conv2D(128, 3)(discriminator_input)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Conv2D(128, 4, strides=2)(dis)
    dis = layers.LeakyReLU()(dis)
    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.4)(dis)
    dis = layers.Dense(1, activation='sigmoid')(dis)
    discriminator = keras.models.Model(discriminator_input, dis)
    # use multiple gpus if available
    try:
        discriminator = multi_gpu_model(discriminator, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    discriminator.summary()
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    discriminator.trainable = False
    return discriminator


def build_gan(generator, discriminator, latent_dim):
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    # use multiple gpus if available
    try:
        gan = multi_gpu_model(gan, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    gan_optimizer = keras.optimizers.RMSprop(
        lr=0.0004, clipvalue=1.0, decay=1e-8
    )
    gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")
    return gan


def build_gan_gen2(generator, discriminator, latent_dim):
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    # use multiple gpus if available
    try:
        gan = multi_gpu_model(gan, gpus=2)
    except:
        print("Can't use multiple gpus!")
        pass
    gan.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    return gan


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
    images, labels = lmdb.read_many_lmdb(image_id_start=train_size + 1,
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
    width = 288
    height = 288
    channels = 3
    latent_dim = 150
    epochs = 20000
    batch_size = 30
    number_images = 5515

    lmdb_path = '/home/steffen/neuronart/scripts/dataProcessing/data/lmdb_5'
    image_path = 'images6'
    model_path = 'portrait1'
    lmdb_size = 200000 * 7500 * 10

    # build gan
    generator = build_generator_gen2(latent_dim=latent_dim, channels=channels)
    discriminator = build_discriminator_gen2(height=height, width=width, channels=channels)
    gan = build_gan_gen2(generator=generator, discriminator=discriminator, latent_dim=latent_dim)

    # init progressbar
    progress_bar = Progbar(target=epochs)

    start = 1
    for step in range(epochs):
        # get actual image batch and convert image values to float
        real_images = get_train_data_batch_images(start, batch_size, lmdb_size, lmdb_path)
        real_images = np.asarray(real_images).astype('float32') / 255.

        # generate new images for discriminator training
        random_values = np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(random_values)

        # train discriminator
        mixed_images = np.concatenate([generated_images, real_images])
        mixed_labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        mixed_labels += 0.05 * np.random.random(mixed_labels.shape)
        d_loss = discriminator.train_on_batch(mixed_images, mixed_labels)

        # train gan
        random_values = np.random.normal(size=(batch_size, latent_dim))
        true_labels = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(random_values, true_labels)

        # update progressbar
        progress_bar.update(step)

        # increment batch counter
        start += 1

        # reset batch counter if all images are processed
        if start > number_images / batch_size - batch_size:
            start = 1

        # output losses and generated image
        if step % 100 == 0:
            # save models as json
            gan_json = gan.to_json()
            generator_json = generator.to_json()
            discriminator_json = discriminator.to_json()

            with open(model_path + "/gan_model.json", "w") as json_file:
                json.dump(gan_json, json_file)

            with open(model_path + "/generator_model.json", "w") as json_file:
                json.dump(generator_json, json_file)

            with open(model_path + "/discriminator_model.json", "w") as json_file:
                json.dump(discriminator_json, json_file)

            # save weights
            gan.save_weights(model_path + '/gan_weights.h5')
            generator.save_weights(model_path + '/generator_weights.h5')
            discriminator.save_weights(model_path + '/discriminator_weights.h5')

            # print losses
            print('Verlust Discriminator: ' + str(d_loss))
            print('Verlust Gegner: ' + str(a_loss))

            # save generated images of the actual epoch
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(image_path, 'generated_artwork' + str(step) + '.png'))
            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(image_path, 'real_artwork' + str(step) + '.png'))
