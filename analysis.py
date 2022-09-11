"""
ML Project: Classification of Handwritten Digits Project
"""
import numpy as np
from tensorflow import keras


def main():
    # 1. load the data into the program (load the MNIST dataset)

    # x_train = the `features array`, contain brightnesses of the pixels (the images features.)
    #  A three-dimensional array (60000, 28, 28), with 60000 elements and each element of
    #   x_train is a 28x28 image

    # y_train = the `target array`, contain classes (digits to predict). A 1D array of digit labels
    #   (the integers from 0 to 9) with 60000 elements

    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

    # we need to convert each image into one dimension array (flatten the image)
    #   so, we reshape a 28x28 image (a 2D array) to a 1D array of 784 elements (28x28)

    # 2. reshape the features array to a 2D array (n x m) with all the images on it
    #   n rows (number images in the dataset) and m columns (number of pixels in each image)
    rows = 60000
    cols = 28 * 28

    reshaped_img_array = np.reshape(x_train, (rows, cols))

    # print("Classes:", np.unique(y_train))
    # print("Features' shape:", np.shape(reshaped_img_array))
    # print("Target's shape:", np.shape(y_train))
    # print(f"min: {reshaped_img_array.min()}, max: {reshaped_img_array.max()}")


if __name__ == '__main__':
    main()