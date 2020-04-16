"""
A simple collection of utilities used in both the iris and vowel tasks.

Started 16.4.2020.
"""

import numpy as np


def visualize_confusion_matrix(confusion_matrix: np.ndarray, categories: list):
    """
    A simple tool to visualize the confusion matrix and error rate of a tested classifier.
    :param confusion_matrix: The confusion matrix, dimension nxn
    :param categories: Ordered list of classes, dimension n
    """
    diagonal_sum = 0
    for i in range(len(confusion_matrix)):
        diagonal_sum += confusion_matrix[i, i]
    error_rate = 1 - diagonal_sum / np.sum(confusion_matrix)

    print("\nResulting in the following confusion matrix:")
    print("Classified as:", end="")
    for item in categories:
        print(f"\t\t{item}", end="")
    print()
    for x in range(len(confusion_matrix)):
        print(f"From class {categories[x]}", end="")
        for y in range(len(confusion_matrix)):
            print(f"\t\t{int(confusion_matrix[x, y])}", end="")
        print()
    print(f"\nThis resulted in an error rate of {100 * error_rate:.1f}%.")
