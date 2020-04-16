"""
A script meant to run the vowel task.

Started 16.4.2020.
"""

from source import *


def task1():
    ###############################################################
    # Task 1 a) - c)
    print("\n\nRunning vowel task part 1\n\n")

    all_data = Dataset()
    training_set, verification_set = all_data.split_at_index(70)

    models_full = single_gaussian_model(training_set, False)
    models_diagonal = single_gaussian_model(training_set, True)

    print("Verifying full model on training set:")
    training_set.verify_gaussian_model(models_full)
    print("\nVerifying full model on verification set:")
    verification_set.verify_gaussian_model(models_full)

    print("\nVerifying diagonal model on training set:")
    training_set.verify_gaussian_model(models_diagonal)
    print("\nVerifying diagonal model on verification set:")
    verification_set.verify_gaussian_model(models_diagonal)


if __name__ == "__main__":
    task1()
