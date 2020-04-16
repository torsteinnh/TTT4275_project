"""
A python script containing utilities used for the vowel task.

Started 15.4.2020.
"""

from __future__ import annotations

from scipy.stats import multivariate_normal
from source.general_utilities import *


class Dataset:
    vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']

    def __init__(self, path: str = "datasett/Wovels/vowdata_nohead.dat", features=(range(7, 16)), sep=" ", typelist=None, dataset=None):
        """
        Initiates the dataset. Reads the data file and selects only the wanted features.
        :param typelist: Alternative type list, must be in pair with dataset
        :param dataset: Alternative data, must be in pair with typelist
        :param path: The path to the datafile to be read
        :param features: The indices at which the wanted features can be found. Remember python is 0 indexed
        :param sep: If the separator somehow changes this is where to fix it
        """
        if typelist is not None and dataset is not None:
            self.internal_types = np.array(typelist)
            self.internal_data = np.array(dataset)
        else:

            with open(path) as file:
                rawdata = file.readlines()
            correctrawdata = list()
            for line in rawdata:
                newline = list()
                splitline = line.split(sep)
                for item in splitline:
                    if item != "":
                        newline.append(item)
                correctrawdata.append(newline)

            self.internal_data = np.zeros((len(rawdata), len(features)), int)
            self.internal_types = ["m01ae"] * len(rawdata)

            for index, line in enumerate(correctrawdata):
                self.internal_types[index] = line[0]
                for featurenumber, feature in enumerate(features):
                    self.internal_data[index, featurenumber] = int(line[feature])

    def _query(self, feature: str) -> np.ndarray:
        """
        A small method to query the internal data sets for types only matching those in the features list
        :param feature: A string for witch to search
        :return: A list of indices in the internal dataset containing the wanted features
        """
        return np.flatnonzero(np.core.defchararray.find(self.internal_types, feature) != -1)

    def __repr__(self) -> str:
        """
        As expected from a __repr__ method, google it.
        :return: The string representing the instance
        """
        return_string = str()

        return_string += f"Representation of dataset with {len(self.internal_types)} elements:\n"
        return_string += f"List of categories:\t{self.internal_types}\n"
        return_string += f"First and last 5 features:\n"
        for i in range(5):
            return_string += f"\t{self.internal_data[i]}\n"
        return_string += f"\t...\n"
        for i in range(4, -1, -1):
            return_string += f"\t{self.internal_data[i]}\n"
        return_string += "For more information, use debugger."

        return return_string

    def split_at_index(self, index, vowellist: list = None) -> (Dataset, Dataset):
        """
        Splits the dataset by category at an index, returns two new data sets.
        :param vowellist: Alternative list of vowels to search for
        :param index: The index at which to split each vowel
        :return: A tuple of data sets, before and after index
        """
        if not vowellist:
            vowellist = Dataset.vowels
        ts_indices, vs_indices = list(), list()

        for vowel in vowellist:
            allindices = self._query(vowel)
            ts_indices.extend(allindices[:index])
            vs_indices.extend(allindices[index:])

        t_newdata = np.array([self.internal_data[i] for i in ts_indices])
        t_newcat = np.array([self.internal_types[i] for i in ts_indices])
        v_newdata = np.array([self.internal_data[i] for i in vs_indices])
        v_newcat = np.array([self.internal_types[i] for i in vs_indices])

        ts = Dataset(typelist=t_newcat, dataset=t_newdata)
        vs = Dataset(typelist=v_newcat, dataset=v_newdata)
        return ts, vs

    def mean_covariance(self, diag: bool = False) -> (np.ndarray, np.ndarray):
        """
        Finds the mean and covariance matrix of the dataset.
        :param diag: Bool indicating if only the diagonal of the covariance matrix should be used
        :return: The mean vector and covariance matrix of the dataset
        """
        mean = np.zeros((len(self.internal_data[0])))
        for index in range(len(self.internal_data[0])):
            temprow = list()
            for number in self.internal_data[:, index]:
                if number != 0:
                    temprow.append(number)
            mean[index] = np.mean(temprow)

        mean = self.internal_data.mean(axis=0)

        difference = self.internal_data - mean
        covariance = difference.T.dot(difference) / len(self.internal_data)

        if diag:
            covariance = np.diag(np.diag(covariance))

        return mean, covariance

    def verify_gaussian_model(self, models: dict, visualize: bool = True) -> np.ndarray:
        """
        Verifies the given dataset given a list of models.
        Works for all list of models where the elements has a method pdf that is vectorized.
        :param models: A list of probability models for each class
        :param visualize: A bool indicating if the results should be printed
        :return: The confusion matrix from the verification process
        """
        vowel_probabilities = np.zeros((len(Dataset.vowels), len(self.internal_data)))
        for index, vowel in enumerate(Dataset.vowels):
            vowel_probabilities[index] = models[vowel].pdf(self.internal_data)
        predicted_vowel_indices = np.argmax(vowel_probabilities, axis=0)

        confusion_matrix = np.zeros((len(Dataset.vowels), len(Dataset.vowels)))
        for instance_index in range(len(self.internal_data)):
            target_index = Dataset.vowels.index(self.internal_types[instance_index][3:5])
            predicted_index = predicted_vowel_indices[instance_index]
            confusion_matrix[target_index, predicted_index] += 1

        if visualize:
            visualize_confusion_matrix(confusion_matrix, Dataset.vowels)

        return confusion_matrix


def single_gaussian_model(trainingset: Dataset, diagonal: bool = False) -> dict:
    """
    Creates a list of gaussian models for the vowels in the training set
    :param trainingset: The training set used to make the models
    :param diagonal: A bool indicating if only the diagonal of the covariance matrix should be used
    :return: A list of the models in the same order as the vowels appear in the dataset
    """
    models = dict()

    for vowel in Dataset.vowels:
        training_slice, _ = trainingset.split_at_index(None, [vowel])
        vowel_mean, vowel_covariance = training_slice.mean_covariance(diagonal)
        vowel_model = multivariate_normal(mean=vowel_mean, cov=vowel_covariance)
        models[vowel] = vowel_model

    return models
