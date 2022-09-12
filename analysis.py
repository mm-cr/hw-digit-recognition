"""
ML Project: Classification of Handwritten Digits Project
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras


def main():
    # 1. load the data into the program (load the MNIST dataset)

    # x_train_features: the `features array` contain brightnesses of the pixels-the images features
    #  A three-dimensional array (60000, 28, 28), with 60000 elements and each element of
    #   x_train is a 28x28 image

    # y_train_target: the `target array`, contain classes (digits to predict).
    #   A 1D array of digit labels (the integers from 0 to 9) with 60000 elements

    (x_train_features, y_train_target), (_, _) = keras.datasets.mnist.load_data()

    # we need to convert each image into one dimension array (flatten the image)
    #   so, we reshape a 28x28 image (a 2D array) to a 1D array of 784 elements (28x28)

    # 2. reshape the features array to a 2D array (n x m) with all the images on it
    #   n rows (number images in the dataset) and m columns (number of pixels in each image)
    rows = 60000  # number images
    cols = 28 * 28  # number of pixels

    reshaped_x_train = np.reshape(x_train_features, (rows, cols))

    # 3. train and test set splitting with train_test_split() from sklearn
    #   x_train: the features' train set
    #   x_test: the features' test set
    #   y_train: a target variable from the train set
    #   y_test: a target variable from the test set

    rows_limit = 6000  # we'll limit the sample to 6000
    seed = 40  # for random shuffling of the rows before the split

    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(
       reshaped_x_train[:rows_limit], y_train_target[:rows_limit], test_size=0.3, random_state=seed
    )

    # print("x_train shape:", x_train_set.shape)
    # print("x_test shape:", x_test_set.shape)
    # print("y_train shape:", y_train_set.shape)
    # print("y_test shape:", y_test_set.shape)
    # print("Proportion of samples per class in train set:")
    # print(pd.Series(y_train_set).value_counts(normalize=True))


if __name__ == "__main__":
    main()
