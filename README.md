# Handwritten Digits Recognition

In this project, I train a Machine Learning model to recognize handwritten digits on a picture. This goal is accomplished by using the MNIST digits classification dataset (a dataset containing 60,000 grayscale images of the 10 digits) provided by [Keras](https://keras.io/), training several classification algorithms, and choosing the more accurate with the help of sklearn tools.

## General Information

The objective is to get hands-on experience with the main classification algorithms, learn how to train them, improve their accuracy, and select the best possible model to recognize handwritten numbers from 0 to 9.
The project uses `Numpy`, `Pandas`, and `Scikit-learn` libraries to implement the supervised learning tasks and achieve the multi-class classification.

## Solution Implementation

### Stage 1: Set up the dataset
To start the project, we set up the dataset and pre-process arrays. We also check the general information about the data. The input is the [MNIST](https://keras.io/api/datasets/mnist/) dataset. The output contains the features and target arrays.

#### Steps:

- Import TensorFlow to load the data and Numpy to transform it.
- Load the data into the program. We only use the x_train and y_train data from the set to form the train and test sets. The sets x_train and x_test are the 'features' array: they contain brightnesses of the pixels -a feature. The y_train and y_test sets are the 'target' array: they contain classes -the digits that the models will predict.
- Reshape the features array to a 2D array with n rows (n = number of images in the dataset) and m columns (m = number of pixels in each image.)
- Check the data: the shape of the features and target arrays. Then print the minimum and maximum values of the features array.

### Stage 2: Splitting data into sets
Here we use sklearn to split the data into train and test sets and then verify the class distribution. In this stage, the input is two arrays: features and target; these two arrays contain the MNIST dataset. The output is four arrays. Those arrays are: a features training set; a features testing set; the target variables from the training set; and the target variables from the test set.

#### Steps:

- Select the first 6000 rows of the dataset.
- Set the test set size to 0.3 and the random seed to 40 (so we can reproduce the output.)
- Check that the set is balanced after splitting by printing the new data shapes and the sample's proportions of the classes in the training set.

### Stage 3: Start training the models using default settings
This stage includes exploring the available sklearn classification algorithms, training the models with default settings, and comparing results. The objective is to find the most suitable algorithm that accurately identifies handwritten numbers.

The algorithms evaluated are: *K-nearest Neighbors**, Decision Tree*, *Logistic Regression*, and *Random Forest*. These four classifier algorithms are trained, and in the following stages, their performances are improved. An accuracy metric is used to evaluate the models.

### Stage 4: Data pre-processing 
In this part, the model's performance is improved through data pre-processing. Normalization affects the accuracy, so in this stage, we experiment with the normalization scales (using the normalizer from sklearn.)

### Stage 5: Hyperparameter tuning 
The model's performance is further improved by fine-tuning the hyperparameters, using `GridSearchCV`, a sklearn tool to achieve this aim. After this, we can keep improving the implementation by manually playing with the parameter's values.


## Learning outcomes
Through this project, I trained several Machine Learning models to identify a digit on a picture, learning the different classifier algorithms provided by sklearn. I learned how to train and test these models, how to measure and compare their performance, and how to improve their performance through the implementation of techniques like normalization and hyperparameter tuning.


