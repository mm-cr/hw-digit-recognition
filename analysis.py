"""
Machine Learning: Handwritten Digits Classification
"""

import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore # accuracy scorer implementation
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from tensorflow import keras


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # Fit the model - The training process is done with the help of the fit() method
    model.fit(features_train, target_train)

    # Make a prediction - predict values in future observations (predict() method)
    model.predict(features_test)

    # calculate accuracy and save it to score
    prediction = model.predict(features_test)
    score = accuracy_score(prediction, target_test)

    return round(score, 3)


def normalization(features_set):
    normalize = Normalizer()
    transformed_data = normalize.fit_transform(features_set)
    return transformed_data


def main():
    # 1. load the data into the program (load the MNIST dataset)

    # x_train_features: the `features array` contain brightnesses of the pixels-the images features
    #  A three-dimensional array (60000, 28, 28), with 60000 elements and each element of
    #   x_train is a 28x28 image. -This is the training data-

    # y_train_target: the `target array`, contain classes (digits to predict).
    #   A 1D array of digit labels (the integers from 0 to 9) with 60000 elements -Training labels-

    # We need both arrays because we're working with a supervised model. With unsupervised models,
    #   the training data array is enough (we don't need annotations).

    (x_train_features, y_train_target), (_, _) = keras.datasets.mnist.load_data()

    # we need to convert each image into one dimension array (flatten the image)
    #   so, we reshape a 28x28 image (a 2D array) to a 1D array of 784 elements (28x28)

    # 2. reshape the features array to a 2D array (n x m) with all the images on it
    #   n rows (number images in the dataset) and m columns (number of pixels in each image)
    rows = 60_000  # number images
    cols = 28 * 28  # number of pixels

    reshaped_x_train = np.reshape(x_train_features, (rows, cols))

    # 3. split the train and test sets with train_test_split() from sklearn
    #   x_features_train: the features' train set
    #   x_features_test: the features' test set
    #   y_target_train: a target variable from the train set
    #   y_target_test: a target variable from the test set

    rows_limit = 6000  # we'll limit the sample to 6000
    seed = 40  # for random shuffling of the rows before the split

    x_features_train, x_features_test, y_target_train, y_target_test = train_test_split(
        reshaped_x_train[:rows_limit],
        y_train_target[:rows_limit],
        test_size=0.3,
        random_state=seed,
    )

    # data normalization to improve accuracy
    x_train_norm = normalization(x_features_train)
    x_test_norm = normalization(x_features_test)

    # TESTING MODELS:
    # create model instances
    # models = [KNeighborsClassifier(),
                # DecisionTreeClassifier(random_state=seed),
                # LogisticRegression(solver="liblinear", random_state=seed),
                # RandomForestClassifier(random_state=seed)]
    # results = {}

    # test models
    # for model in models:
    #    results[type(model).__name__] = fit_predict_eval(
    #                                                       model,
    #                                                       x_train_norm,
    #                                                       x_test_norm,
    #                                                       y_target_train,
    #                                                       y_target_test)

    # for model, score in results.items():
    #   print(f'Model: {model}\nAccuracy: {score}\n')

    # sorted_results = sorted(results, key=results.get)

    # best_model = sorted_results[-1]  # get key with max value
    # second_best_mdl = sorted_results[-2]
    # best_score = results[best_model]
    # second_best_scr = results[second_best_mdl]
    # print(f'Best models: {best_model} - {best_score}, {second_best_mdl} - {second_best_scr}' )

    # Hyperparameters:

    # KNeighborsClassifier
    estimator_kn = KNeighborsClassifier()
    param_grid_kn = dict(
        n_neighbors=[3, 4], weights=["uniform", "distance"], algorithm=["auto", "brute"]
    )
    grid_kn = GridSearchCV(estimator_kn, param_grid_kn, scoring="accuracy", n_jobs=-1)
    grid_kn.fit(x_train_norm, y_target_train)

    print("K-nearest neighbours algorithm")
    print("best estimator: " + str(grid_kn.best_estimator_))

    score = fit_predict_eval(
        grid_kn.best_estimator_, x_train_norm, x_test_norm, y_target_train, y_target_test
    )
    print("accuracy: " + str(score))
    print("")

    # RandomForestClassifier
    estimator_rf = RandomForestClassifier(random_state=seed)
    param_grid_rf = dict(
        n_estimators=[300, 500],
        max_features=["sqrt", "log2"],
        class_weight=["balanced", "balanced_subsample"],
    )
    grid_rf = GridSearchCV(estimator_rf, param_grid_rf, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(x_train_norm, y_target_train)

    print("Random forest algorithm")
    print("best estimator: " + str(grid_rf.best_estimator_))

    score = fit_predict_eval(
        grid_rf.best_estimator_, x_train_norm, x_test_norm, y_target_train, y_target_test
    )
    print("accuracy: " + str(score))


if __name__ == "__main__":
    main()
