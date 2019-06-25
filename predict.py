import argparse
import os

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

if __name__ == '__main__':

    #Define input arguments:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--dataset',
        help='Path to the test dataset',
        action='store',
        dest='dataset',
        required=True)
    parser.add_argument(
        '--model',
        help='Path to the saved model',
        action='store',
        dest='model',
        required=True)
    parser.add_argument(
        '--output-predictions',
        help='Path to the output predictions file (.csv)',
        action='store',
        dest='output_predictions',
        required=True)

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    output_predictions = args.output_predictions

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
        validation_split=0)

    test_gen = img_gen.flow_from_directory(
        dataset,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode=None,
        seed=2)

    trained_model = load_model(model)
    filenames = test_gen.filenames
    num_samples = len(filenames)

    predictions = trained_model.predict_generator(test_gen, steps=num_samples, verbose=1)

    LINES = []

    with open('data/class_id_mapping.csv') as f:
        classes = ['filename']
        for line in f.readlines()[1:]:
            class_name = line.split(",")[0]
            classes.append(class_name)
    LINES.append(','.join(classes))

    for idx, _file_path in enumerate(filenames):
        probs = list(map(str, predictions[idx]))
        LINES.append(",".join([os.path.basename(_file_path)] + probs))

    fp = open(output_predictions, "w")
    fp.write("\n".join(LINES))
    fp.close()
