from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
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
    model.add(Dense(units=1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
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


def plot(x,y1,y2,xlabel,ylabel1,ylabel2,loss_file):
    plt.figure(num=None, figsize=(16,12), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(x, y1, 'b', label=ylabel1, linewidth=0.7)
    plt.plot(x, y2, 'g', label=ylabel2, linewidth=0.7)
    plt.legend()
    plt.minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel1)
    plt.ylabel(ylabel2)
    #plt.yticks(np.arange(0, 2.5, 0.1))
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.5', linestyle='--', linewidth=0.5)
    plt.savefig(loss_file)
    plt.close()

def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()

    # model summary
    model_dir = './models'
    if not os.path.exists(model_dir):
		os.mkdir(model_dir)		
    with open(model_dir + '/discriminator.txt','w') as fh:
    	d.summary(print_fn=lambda x: fh.write(x + '\n'))
	with open(model_dir + '/generator.txt','w') as fh:
		g.summary(print_fn=lambda x: fh.write(x + '\n'))
    plot_model(d, to_file=model_dir+'/discriminator.png')
    plot_model(g, to_file=model_dir+'/generator.png')

    val_dir = './val'
    if not os.path.exists(val_dir):
		os.mkdir(val_dir)		

    log_dir = './logs'
    if not os.path.exists(log_dir):
		os.mkdir(log_dir)	

    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    d_losses_epoch = []
    g_losses_epoch = []
    for epoch in range(100):
        d_losses = []
        g_losses = []
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        d_losses_epoch.append(np.mean(d_losses))
        g_losses_epoch.append(np.mean(g_losses))
        print("epoch %03d) d_loss : %f / g_loss : %f" % (epoch, d_losses_epoch[-1], g_losses_epoch[-1]))
        
        # plot losses
        loss_file_name = log_dir + '/epoch-{:03d}.png'.format(epoch)
        plot([x+1 for x in range(epoch+1)],d_losses_epoch,g_losses_epoch,
        	'epoch','discriminator','generator',loss_file_name)

        # save weights
        g.save_weights(model_dir+'/generator.h5', True)
        d.save_weights(model_dir+'/discriminator.h5', True)

        # save results from training
        image = combine_images(generated_images)
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(
            val_dir+"/epoch-{:03d}.png".format(epoch))


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    model_dir = './models'
    g.load_weights(model_dir+'/generator.h5')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
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
