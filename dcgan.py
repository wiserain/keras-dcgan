from keras.models import Sequential
from keras.layers import Reshape, AveragePooling2D, Dense, Dropout, Flatten
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras import initializers
from keras.utils.generic_utils import Progbar
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

import os
from keras.utils import plot_model
import matplotlib.pyplot as plt

##################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def generator_model():
    model = Sequential()

    model.add(Dense(3 * 3 * 384, input_dim=100, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((3, 3, 384)))

    model.add(Conv2DTranspose(192, 5, strides=1, padding='valid'))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(96, 5, strides=2, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(1, 5, strides=2, padding='same',activation='tanh'))
    return model

def discriminator_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),padding='same',strides=2,input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64,3,padding='same', strides=1))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128,3,padding='same', strides=2))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256,3,padding='same', strides=1))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def plot(x,y1,y2,xlabel,ylabel,ylabel1,ylabel2,loss_file):
    plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(x, y1, 'b', label=ylabel1, linewidth=0.7)
    plt.plot(x, y2, 'g', label=ylabel2, linewidth=0.7)
    plt.legend()
    plt.minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.5', linestyle='--', linewidth=0.5)
    plt.savefig(loss_file)
    plt.close()

def train(BATCH_SIZE):

    # saving results to
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    val_dir = './vals'
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    nepoch = 200

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # ganhacks 1: normalize inputs
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = np.expand_dims(X_train, axis=-1)

    lr = 2e-4

    # generator
    g = generator_model()
    g_optim = Adam(lr=lr, beta_1=0.5)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)

    # discriminator
    d = discriminator_model()

    # gan: g+d
    gan_optim = Adam(lr=lr, beta_1=0.5)
    gan = generator_containing_discriminator(g, d)
    gan.compile(loss='binary_crossentropy', optimizer=gan_optim)

    # discriminator compile
    d_optim = Adam(lr=lr, beta_1=0.5)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)


    # model summary
    with open(model_dir + '/discriminator.txt','w') as fh:
    	d.summary(print_fn=lambda x: fh.write(x + '\n'))
	with open(model_dir + '/generator.txt','w') as fh:
		g.summary(print_fn=lambda x: fh.write(x + '\n'))
    plot_model(d, to_file=model_dir+'/discriminator.png')
    plot_model(g, to_file=model_dir+'/generator.png')


    d_losses_epoch = []
    g_losses_epoch = []
    for epoch in range(1,nepoch+1):

        d_losses = []
        g_losses = []

        num_batches = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=num_batches)

        for index in range(num_batches):

            # ganhacks 3: sample from Gaussian
            noise = 0.5 * np.random.normal(0, 1, size=[BATCH_SIZE, 100])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            # generage fake mnist images
            generated_images = g.predict(noise, verbose=0)

            # train discriminator
            X = np.concatenate((image_batch, generated_images))
            yD = [0.9] * BATCH_SIZE + [0.] * BATCH_SIZE          # hard label
            # yD += 0.05 * np.random.normal(size=(2*BATCH_SIZE,))
            d_loss = d.train_on_batch(X, yD)

            # train generator
            noise = 0.5 * np.random.normal(0, 1, size=[BATCH_SIZE, 100])
            d.trainable = False
            yG = [1.] * BATCH_SIZE
            # yG += 0.05 * np.random.normal(size=(BATCH_SIZE,))
            g_loss = gan.train_on_batch(noise, yG)
            d.trainable = True

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            progress_bar.update(index + 1)

        d_losses_epoch.append(np.mean(d_losses))
        g_losses_epoch.append(np.mean(g_losses))
        print("Epoch {:03d}/{:03d} - d_loss: {} -  g_loss: {}".format(epoch, nepoch, d_losses_epoch[-1], g_losses_epoch[-1]))

        # plot losses
        loss_file_name = log_dir + '/epoch-{:03d}.png'.format(epoch)
        plot([x for x in range(1,epoch+1)],d_losses_epoch,g_losses_epoch,
        	'epoch','loss','discriminator','generator',loss_file_name)

        # save weights
        g.save_weights(model_dir+'/generator.h5', True)
        d.save_weights(model_dir+'/discriminator.h5', True)

        # save results from training
        image = combine_images(generated_images)
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(
            val_dir+"/epoch-{:03d}.png".format(epoch))


def generate(BATCH_SIZE, nice=False):
    model_dir = './models'
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="Adam")
    g.load_weights(model_dir+'/generator.h5')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="Adam")
        d.load_weights(model_dir+'/discriminator.h5')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
