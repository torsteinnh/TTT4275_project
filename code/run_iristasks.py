"""
A script meant to run the iris task.

Started 13.4.2020.
"""

import sys
sys.path.append('code/')

from iris_utilities import *
import numpy as np


def task1():
    print("Starting task 1 on iris classification\n\n")
    ts, vs = load_iris()

    print("Training the classifier on first 30 instances:")
    w = train_classifier(ts)

    print("\nVerifying the classifier on first 30 instances:")
    verify_classifier(w, ts)

    print("\nVerifying the classifier on the last 20 instances:")
    verify_classifier(w, vs)


    ###############################################################
    # Task 1 d)

    ts, vs = load_iris(np.s_[:, 20:], np.s_[:, :20])

    print("Training the classifier on last 30 instances:")
    w = train_classifier(ts)

    print("\nVerifying the classifier on last 30 instances:")
    verify_classifier(w, ts)

    print("\nVerifying the classifier on the first 20 instances:")
    verify_classifier(w, vs)


if __name__ == "__main__":
    task1()
