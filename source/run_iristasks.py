"""
A script meant to run the iris task.

Started 13.4.2020.
"""

from source import *
import numpy as np
import matplotlib.pyplot as plt


def task1():
    ###############################################################
    # Task 1 a) - c)
    print("Starting task 1 on iris classification\n\n")
    ts, vs = load_iris()

    print("Training the classifier on first 30 instances:")
    w = train_linear_classifier(ts)

    print("\nVerifying the classifier on first 30 instances:")
    verify_linear_classifier(w, ts)

    print("\nVerifying the classifier on the last 20 instances:")
    verify_linear_classifier(w, vs)

    ###############################################################
    # Task 1 d)

    ts, vs = load_iris(np.s_[:, 20:], np.s_[:, :20])

    print("Training the classifier on last 30 instances:")
    w = train_linear_classifier(ts)

    print("\nVerifying the classifier on last 30 instances:")
    verify_linear_classifier(w, ts)

    print("\nVerifying the classifier on the first 20 instances:")
    verify_linear_classifier(w, vs)


def task2(path: str = None):
    ###############################################################
    # Task 2 a)
    alldata, _ = load_iris(np.s_[:])

    print("\n\nStarting task 2 on iris classification\n\n")
    print("Plotting histograms for different features.")
    if path:
        print(f"Histograms saved to {path}")
    else:
        print("Remember to close the interactive view of figure.")

    fig, axs = plt.subplots(2, 2)
    for trait in range(4):
        bins = np.linspace(np.amin(alldata[:, :, trait]), np.amax(alldata[:, :, trait]), int(len(alldata[0]) / 2))
        for category in range(3):
            axs[trait // 2, trait % 2].hist(alldata[category, :, trait], bins, density=True, alpha=0.5, label=f"Class {category + 1}")
        axs[trait // 2, trait % 2].legend()
        axs[trait // 2, trait % 2].set_title(f"Histogram of feature {trait + 1}")

    if path:
        plt.savefig(path)
    else:
        plt.show()

    ###############################################################
    # Task 2 b)
    ts = alldata[:, :30, :]
    vs = alldata[:, 30:, :]

    print("\n\nRemoving second feature:\n\nTraining:")
    tts = select_iris_features(ts, [0, 2, 3])
    tvs = select_iris_features(vs, [0, 2, 3])
    w = train_linear_classifier(tts)
    print("Verifying og verification set")
    verify_linear_classifier(w, tvs)

    print("\n\nRemoving first and second feature:\n\nTraining:")
    tts = select_iris_features(ts, [2, 3])
    tvs = select_iris_features(vs, [2, 3])
    w = train_linear_classifier(tts)
    print("Verifying og verification set")
    verify_linear_classifier(w, tvs)

    print("\n\nRemoving first, second and third feature:\n\nTraining:")
    tts = select_iris_features(ts, [3])
    tvs = select_iris_features(vs, [3])
    w = train_linear_classifier(tts)
    print("Verifying og verification set")
    verify_linear_classifier(w, tvs)


if __name__ == "__main__":
    task1()
    task2()
