import data_util
import math_util
from pca import PCA
from lda import LDA
import numpy as np
import matplotlib.pyplot as plt


def preprocess():
    data = data_util.load_data("theory.csv")
    print("Data:\n", data)

    X = data[:, :-1]
    y = data[:, -1]

    print("\nX:\n", X)
    print("\ny:\n", y)

    means, stds = math_util.calculate_feature_mean_std(X)
    x_zscored = math_util.z_score_data(X, means, stds)

    print("\nMeans:", means)
    print("\nStd:", stds)
    print("\nX-Zscored:\n", x_zscored)

    return x_zscored, y


def plot():
    print("\nTHEORY - DATA")

    # Load and Zscore Data
    X, y = preprocess()

    # Title
    title = "Class1 vs. Class2"

    # Name of File
    filename = "theory_" + title.lower().replace(" ", "_") + ".png"

    # Title of the Plot
    plt.title(title)

    # Set our X and Y Labels
    plt.xlabel('Class1', fontsize=14)
    plt.ylabel('Class2', fontsize=14)

    # Split data based on class
    X_1 = X[y == 1]
    X_2 = X[y == 2]

    # Plot our data
    plt.scatter(X_1[:, 0], X_1[:, 1], marker="s")
    plt.scatter(X_2[:, 0], X_2[:, 1], marker="o")

    # Save and show our plot
    plt.savefig(filename)
    plt.show()


def pca():
    print("\nTHEORY - PCA")

    # Load and Zscore Data
    X, y = preprocess()

    model = PCA(num_components=1)
    eigenvectors = model.train_model(X)
    print("\nEigenvectors:\n", eigenvectors)

    z = model.evaluate_model(X, eigenvectors)
    print("\nZ:\n", z)

    # Title
    title = "PCA 1D"

    # Name of File
    filename = "theory_" + title.lower().replace(" ", "_") + ".png"

    # Title of the Plot
    plt.title(title)

    # Set our X and Y Labels
    plt.xlabel('Class1', fontsize=14)
    plt.ylabel('Class2', fontsize=14)

    # Plot our data
    zero_y = np.zeros((z.shape))
    plt.scatter(z[y == 1], zero_y[y == 1], marker="s")
    plt.scatter(z[y == 2], zero_y[y == 2], marker="o")

    # Save and show our plot
    plt.savefig(filename)
    plt.show()


def lda():
    print("\nTHEORY - LDA")

    # Load and Zscore Data
    X, y = preprocess()

    model = LDA(num_components=1)
    eigenvectors = model.train_model(X, y)
    print("\nEigenvectors:\n", eigenvectors)

    z = model.evaluate_model(X, eigenvectors)
    print("\nZ:\n", z)

    # Title
    title = "LDA 1D"

    # Name of File
    filename = "theory_" + title.lower().replace(" ", "_") + ".png"

    # Title of the Plot
    plt.title(title)

    # Set our X and Y Labels
    plt.xlabel('Class1', fontsize=14)
    plt.ylabel('Class2', fontsize=14)

    # Plot our data
    zero_y = np.zeros((z.shape))
    plt.scatter(z[y == 1], zero_y[y == 1], marker="s")
    plt.scatter(z[y == 2], zero_y[y == 2], marker="o")

    # Save and show our plot
    plt.savefig(filename)
    plt.show()


plot()
pca()
lda()