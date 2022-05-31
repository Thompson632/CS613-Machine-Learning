import numpy as np
import matplotlib.pyplot as plt


def plot_pca_scatterplot(title, data):
    '''
    Plots the computed principal components data onto a MatplotLib 
    scatter plot.

    :param title: The title to be added to the plot and used 
    as the filename when saving the image.
    :param data: The data that is to be scattered onto the plot

    :return None
    '''
    # Name of File
    filename = title.lower().replace(" ", "_") + ".png"

    # Title of the Plot
    plt.title(title)

    # Set our X and Y Labels
    plt.xlabel('PCA1', fontsize=14)
    plt.ylabel('PCA2', fontsize=14)

    # Plot our data
    plt.scatter(data[:, 0], data[:, 1])

    # Save and show our plot
    plt.savefig(filename)
    plt.show()


def plot_pca_scatterplot_overlay(title, pca, wpca):
    '''
    Plots the principal components of our non-whitened and whitened
    data overlayed onto one another. This is used to show a clear 
    distinction between the non-whitened principal components and
    our whitend more spherical principal components.

    :param title: The title to be added to the plot and used 
    as the filename when saving the image.
    :param pca: The non-whitened principal components
    :param wpca: The whitened principal components

    :return None
    '''
    # Name of File
    filename = title.lower().replace(" ", "_") + ".png"

    # Title of the Plot
    plt.title(title)

    # Set our X and Y Labels
    plt.xlabel('PCA1', fontsize=14)
    plt.ylabel('PCA2', fontsize=14)

    # Plot Non-Whitened Data
    plt.scatter(pca[:, 0], pca[:, 1], label='Non-Whitened')
    # Plot Whitened Data
    plt.scatter(wpca[:, 0], wpca[:, 1], label='Whitened')

    # Save and show our plot
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_eigenfaces(title, eigenvectors):
    '''
    Plots the eigenfaces of our eigenvectors.

    :param title: The title to be added to the plot and used
    as the filename when saving the image.
    :param eigenvectors: The eigen vectors needed to reconstruct
    the image

    :return None
    '''
    # Name of File
    filename = title.lower().replace(" ", "_").replace("%", "") + ".png"

    # Title of the Plot
    plt.title(title)

    # Plot the Image
    plt.imshow(eigenvectors.real, interpolation='none', cmap='gray')

    # Save and show our image
    plt.savefig(filename)
    plt.show()