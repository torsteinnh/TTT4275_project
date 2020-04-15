"""
A python script containing utilities used for the vowel task.

Started 15.4.2020.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


class Dataset:
    vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']

    def __init__(self, path: str = "datasett/Wovels/vowdata_nohead.dat", features=(13, 14, 15), sep=" ", typelist=None, dataset=None):
        """
        Initiates the dataset. Reads the data file and selects only the wanted features.
        :param typelist: Alternative type list, must be in pair with dataset
        :param dataset: Alternative data, must be in pair with typelist
        :param path: The path to the datafile to be read
        :param features: The indices at which the wanted features can be found. Remember python is 0 indexed
        :param sep: If the separator somehow changes this is where to fix it
        """
        if typelist is not None and dataset is not None:
            self.internal_types = typelist
            self.internal_data = dataset
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

    def query(self, features: list) -> Dataset:
        """
        A small method to query the internal data sets for types only matching those in the features list
        :param features: A list of strings, these are the features searched for
        :return: A new dataset with only the wanted features
        """
        return_data = np.zeros(self.internal_data.shape, int)
        return_types = list()

        found = 0
        for index, instancetype in enumerate(self.internal_types):
            for selections in features:
                if selections in instancetype:
                    return_data[found] = self.internal_data[index]
                    return_types.append(instancetype)
                    found += 1
                    break

        return Dataset(typelist=return_types, dataset=return_data)

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

    def split_at_index(self, index: int) -> (Dataset, Dataset):
        """
        Splits the dataset at an index, returns two new data sets.
        :param index: The index at which to split
        :return: A tuple of data sets, before and after index
        """
        ts = Dataset(typelist=self.internal_types[:index], dataset=self.internal_data[:index])
        vs = Dataset(typelist=self.internal_types[index:], dataset=self.internal_data[index:])
        return ts, vs

    def mean_covariance(self, diag: bool = False) -> (np.ndarray, np.ndarray):
        """
        Finds the mean and covariance matrix of the dataset.
        :param diag: Bool indicating if only the diagonal of the covariance matrix should be used
        :return: The mean vector and covariance matrix of the dataset
        """
        mean = self.internal_data.mean(axis=0)
        difference = self.internal_data - mean
        covariance = difference.T.dot(difference) / len(self.internal_data)

        if diag:
            covariance = np.diag(np.diag(covariance))

        return mean, covariance


def single_gaussian_model(trainingset: Dataset, visualizationset: Dataset, visualize: bool = True):
    pass


testcase = Dataset()
print(testcase)
print("\n")
print(testcase.query(["ae", "iy"]))
print()
print(testcase.mean_covariance(True))
