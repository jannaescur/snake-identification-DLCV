import argparse
import os
import warnings

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications import ResNet50, VGG16, NASNetMobile, InceptionResNetV2


def get_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        return Adam(lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        warnings.warn(
            f"Learning rate setting not supported by {optimizer} optimizer",
            UserWarning)
        return optimizer

def create_inception_resnet_v2_model():
    inception_resnetv2 = InceptionResNetV2(
        include_top=False, weights='imagenet', input_shape=(299,299,3))
    inception_resnetv2_out = inception_resnetv2[-1].output
    inception_resnetv2_out = GlobalAveragePooling2D()(inception_resnetv2_out)
    x = Dense(512, activation='relu')(inception_resnetv2_out)
    x = Dense(256, activation='relu')(x)
    x = Dense(45)(x)
    x = Activation(tf.nn.softmax)(x)

    model = Model(inception_resnetv2.input, x)

    return model


def create_nasnet_model():
    NASnet = NASNetMobile(
        include_top=False, weights='imagenet', input_shape= (224, 224, 3))
    nasnet_out = NASnet.layers[-1].output
    nasnet_out = GlobalAveragePooling2D()(nasnet_out)
    x = Dense(512, activation='relu')(nasnet_out)
    x = Dense(256, activation='relu')(x)
    x = Dense(45)(x)
    x = Activation(tf.nn.softmax)(x)

    model = Model(NASnet.input, x)

    return model


def create_resnet_model():
    resnet = ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    resnet_out = resnet.layers[-1].output
    resnet_out = GlobalAveragePooling2D()(resnet_out)
    x = Dense(512, activation='relu')(resnet_out)
    x = Dense(256, activation='relu')(x)
    x = Dense(45)(x)
    x = Activation(tf.nn.softmax)(x)

    model = Model(resnet.input, x)

    return model


def create_vgg_model():
    vgg = VGG16(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vgg_out = vgg.layers[-1].output
    vgg_out = GlobalAveragePooling2D()(vgg_out)
    x = Dense(512, activation='relu')(vgg_out)
    x = Dense(256, activation='relu')(x)
    x = Dense(45)(x)
    x = Activation(tf.nn.softmax)(x)

    model = Model(vgg.input, x)

    return model


if __name__ == '__main__':

    #Define input arguments:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--dataset',
        help='Path to the dataset',
        action='store',
        dest='dataset',
        required=True)
    parser.add_argument(
        '--checkpoints',
        help='Folder where checkpoints will be stored',
        action='store',
        dest='checkpoints',
        required=True)
    parser.add_argument(
        '--experiment-name',
        help='Name of the experiment',
        action='store',
        dest='experiment_name',
        required=True)
    parser.add_argument(
        '--model-name',
        help='Which model to use. Options: resnet, vgg, nasnet, inception_resnet_v2',
        action='store',
        dest='model_name',
        default='vgg')
    parser.add_argument(
        '--epochs',
        help='Number of epochs',
        type=int,
        action='store',
        dest='epochs',
        default=100)
    parser.add_argument(
        '--batch-size',
        help='Size of the batch',
        type=int,
        action='store',
        dest='batch_size',
        default=16)
    parser.add_argument(
        '--optimizer',
        help='Name of the optimizer to be used',
        action='store',
        dest='optimizer',
        default='sgd')
    parser.add_argument(
        '--lr',
        help='Learning rate of your optimizer',
        type=float,
        action='store',
        dest='lr',
        default=0.001)

    args = parser.parse_args()
    dataset = args.dataset
    checkpoints = args.checkpoints
    experiment_name = args.experiment_name
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = args.optimizer
    lr = args.lr

    if model_name == 'resnet':
        model = create_resnet_model()
        target_size = (224, 224)
    elif model_name == 'vgg':
        model = create_vgg_model()
        target_size = (224, 224)
    elif model_name == 'nasnet':
        model = create_nasnet_model()
        target_size = (224, 224)
    elif model_name == 'inception_resnet_v2':
        model = create_inception_resnet_v2_model()
        target_size = (299, 299)

    optimizer = get_optimizer(optimizer, lr)

    model.summary()

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    img_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=True,
        featurewise_std_normalization=False,
        samplewise_std_normalization=True,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=0,
        preprocessing_function=preprocess_input,
        data_format=None,
        validation_split=0.2)

    train_gen = img_gen.flow_from_directory(
        dataset,
        target_size=target_size,
        batch_size=batch_size,
        seed=2,
        subset='training')

    val_gen = img_gen.flow_from_directory(
        dataset,
        target_size=target_size,
        batch_size=batch_size,
        seed=2,
        subset='validation')

    early_stopping = EarlyStopping(patience=10)

    checkpointer = ModelCheckpoint(
        os.path.join(checkpoints, experiment_name, 'best.h5'),
        verbose=1,
        save_best_only=True)

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=0,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0)

    tb = TensorBoard(
        log_dir=os.path.join(checkpoints, experiment_name),
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='batch')

    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpointer, reduce_on_plateau, tb],
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size)

    model.save(os.path.join(checkpoints, experiment_name, 'final.h5'))
