"""
A python script containing utilities used for the iris task.

Started 13.4.2020.
"""

import numpy as np


def load_iris(trainingslice=np.s_[:, :30], verificationslice=np.s_[:, 30:], path="datasett/Iris_TTT4275/iris.data") -> (np.ndarray, np.ndarray):
    """
    Loads data from iris set, formats it correctly and returns the training and verification as asked for.
    Dataset is a matrix on the form [category, instance, feat[1..4]]
    :return:
    :param trainingslice: Optional slice for training set, generate with np.s_
    :param verificationslice: Optional slice for verification set, generate with np.s_
    :param path: Optional alternate path to iris dataset
    """

    raw_data = np.genfromtxt(path, dtype=str, delimiter=",")
    all_data = np.empty((3, 50, 4))

    read_indices = [0, 0, 0]

    for instance in raw_data:
        type_index = 0 if instance[4] == "Iris-setosa" else 1 if instance[4] == "Iris-versicolor" else 2
        all_data[type_index, read_indices[type_index]] = instance[:4]

        read_indices[type_index] += 1

    # It seems like removing the mean and standard deviation decreases the accuracy of the classifier.
    # for feature in range(4):
    #     all_data[:, :, feature] -= np.mean(all_data[:, :, feature])
    #     all_data[:, :, feature] /= np.std(all_data[:, :, feature])

    return all_data[trainingslice], all_data[verificationslice]


def sigmoid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Implementation of the sigmoid function used by the linear classifier
    :param x: The instance of the problem to classify
    :param w: The matrix derived such that Wx signifies the correct class of x
    :return: A vector with the highest value for the class x belongs to
    """
    return 1 / (1 + np.e ** (-np.matmul(w, x)))


def mse_gradient(w: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of the MSE as a function of the classifier W and the training set ts.
    Follows equation 3.22 in the compendium.
    :param w: The matrix derived such that Wx signifies the correct class of x
    :param ts: The training set to find the MSE of
    :return: The gradient of the MSE for the training set classified with W
    """
    grad = np.zeros((len(ts[0, 0]), 3))

    for t, targetc in enumerate(ts):
        targetv = np.array([0, 0, 0])
        targetv[t] = 1

        for instance in targetc:
            g = sigmoid(instance, w)
            x = np.array([instance])

            grad = grad + ((g - targetv).dot(g.dot(1 - g))) * x.T

    return grad


def train_classifier(ts: np.ndarray, iterations: int = 1000, alpha: float = 0.0005, progress: bool = True) -> np.ndarray:
    """
    Trains the classifier W given the training set ts
    :param ts: The training set to be used.
    :param iterations: The amount of times to adjust the classifier
    :param alpha: The step factor
    :param progress: A bool indicating if the status of the training should be printed
    :return: The weighted matrix W used in the linear classifier
    """
    w = np.zeros((3, len(ts[0, 0])))

    for n in range(iterations):
        w -= alpha * mse_gradient(w, ts).T

        if progress and (100 * n / iterations) % 10 == 0:
            print(f"\rProgress passed {100 * n / iterations}%", end="")

    if progress:
        print("\rFinished training")

    return w


def verify_classifier(w: np.ndarray, vs: np.ndarray, visualize: bool = True) -> np.ndarray:
    """
    Simple verification of a classifier, finds the confusion matrix after testing all input.
    :param w: The weighed classifier matrix
    :param vs: The verification dataset
    :param visualize: A bool indicating if the results should be printed nicely
    :return: The confusion matrix from the verification
    """
    confusion_matrix = np.zeros((3, 3))

    for t, target_class in enumerate(vs):
        for instance in target_class:
            prediction = np.argmax(np.matmul(w, instance))
            confusion_matrix[t, prediction] += 1

    if visualize:
        error_rate = 1 - (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[2, 2])/np.sum(confusion_matrix)

        print("Testing of the following weighted classifier matrix:")
        for x in range(len(w)):
            for y in range(len(w[0])):
                print(f"\t{w[x, y]:.2f}", end="")
            print()
        print("\nResulting in the following confusion matrix:")
        for x in range(3):
            for y in range(3):
                print(f"\t\t{int(confusion_matrix[x, y])}", end="")
            print()
        print(f"\nThis resulted in an error rate of {100 * error_rate:.1f}%.")

    return confusion_matrix


def select_features(data: np.ndarray, features: list) -> np.ndarray:
    """
    Simple tool to select only a few features from a dataset.
    This is not effective at all as it allocates more memory. Do not use lightly.
    :param data: The dataset containing among other the relevant features.
    :param features: A tuple of integers specifying the indices of the wanted features.
    :return: A new dataset containing only the wanted features.
    """
    newdata = np.zeros((len(data), len(data[0]), len(features)))

    features.sort()
    for index, feat in enumerate(features):
        newdata[:, :, index] = data[:, :, feat]

    return newdata
