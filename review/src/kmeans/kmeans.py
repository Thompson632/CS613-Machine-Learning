import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math
import sys

from sklearn import cluster


def distance(pt1, pt2, columns):
    """
    Find the L1 distance between pt1 and pt2. The number of columns in each point must be the same.
    The exact column names must also match. A list of columns must be passed in to only consider those columns when computing distance.
    :param pt1: Source point
    :param pt2: Destination point
    :param columns: A list of columns to use when computing distance, only these columns will be considered
    :return:
    """
    return math.sqrt(sum((pt1[columns] - pt2[columns])**2))


def assign_membership(df, cluster_centers, columns):
    """
    Return a dictionary of clusters to assigned indices
    The key of the dictionary is the index of the cluster in data_indices
    The value is a list of indices of points within the given df
    :param df: DataFrame containing data points
    :param data_indicies: list of indicies which serve as center points
    :param columns: List of column names to consider when computing distance.
    :return:
    """
    membership_df = {ki: [] for ki in range(
        len(cluster_centers))}  # Key = point index, Value = Assigned Cluster
    for i in range(len(df)):
        min_distance = sys.maxsize
        assigned_cluster = -1
        for ki in range(len(cluster_centers)):
            pt = cluster_centers[ki]
            dist = distance(pt, df.iloc[i], columns)
            if(dist < min_distance):
                assigned_cluster = ki
                min_distance = dist

        membership_df[assigned_cluster].append(i)

    return membership_df


def kmeans(dataframe, col_1_index, col_2_index, k, output_path, columns, cluster_colors=['r', 'b']):
    """
    Perform Clustering using K-Means
    :param dataframe:
    :param col_1_index: The index of the first feature to plot
    :param col_2_index: The index of the second feature to plot
    :param k: The k value for k-means
    :param output_path: The path to save the graphs to.
    :param columns: The list of columns to use when computing distance.
    :param cluster_colors: A list of colors to use when displaying clusters in graphs.
    :return:
    """
    print(df.head)
    # Standardize Data
    std_df = zscore(dataframe)

    # Seed Random Number Generator with '0'
    random.seed(0)

    # Randomly select k data instances and use for initial means
    cluster_centers = [std_df.iloc[random.randrange(
        0, len(std_df)-1)] for i in range(k)]

    print("Cluster Centers:\n", cluster_centers)

    # Plot the initial Setup
    plot_setup(cluster_centers, col_1_index, col_2_index, output_path, std_df)
    plt.clf()

    # Do K-Means Algorithm
    # Keep Count of Iterations
    membership = {}
    iteration = 0
    while True:
        plt_name = "InitialAssignment"
        plt_title = "Initial Assignment"

        if iteration > 0:
            plt_name = "Iteration{0}".format(iteration)
            plt_title = "Iteration #{0}".format(iteration)

        # Assign Membership
        membership = assign_membership(std_df, cluster_centers, columns)

        # Plot Cluster Assignments
        plt.clf()
        # plot_iteration(cluster_centers, cluster_colors, col_1_index, col_2_index, k,
        #                membership, output_path, std_df, plt_name, plt_title)

        # Update Center
        # For each 'cluster' compute the mean point
        new_cluster_centers = []
        for cluster_i in range(len(cluster_centers)):
            indices = membership[cluster_i]
            new_cluster_centers.append(std_df.iloc[indices].mean())

        # Compute difference between prior center and new center, if < 1eps, exit
        difference = sys.maxsize
        for cluster_i in range(len(cluster_centers)):
            old_point = cluster_centers[cluster_i]
            new_point = new_cluster_centers[cluster_i]
            difference = (old_point - new_point).max()

        print("Iteration #{0}, Difference: {1}".format(iteration, difference))
        if difference <= np.spacing(1):  # Matlab's Eps == np.spacing(1)
            break

        cluster_centers = new_cluster_centers
        iteration += 1

    # Plot final cluster assignments
    plot_iteration(cluster_centers, cluster_colors, col_1_index, col_2_index, k,
                   membership, output_path, std_df, "FinalAssignment", "Final Cluster Assignments")


def plot_setup(cluster_centers, col_1_index, col_2_index, output_path, std_df):
    # Plot the Data Points
    plt.plot(std_df[std_df.columns[col_1_index]],
             std_df[std_df.columns[col_2_index]], 'rx')
    for cluster_center in cluster_centers:
        plt.plot(cluster_center[cluster_center.index[col_1_index]],
                 cluster_center[cluster_center.index[col_2_index]], 'bo')
    plt.title("Initial Configuration")
    plt.xlabel(std_df.columns[col_1_index])
    plt.ylabel(std_df.columns[col_2_index])
    plt.savefig(os.path.join(output_path, "InitialConfiguration.png"))


def plot_iteration(cluster_centers, cluster_colors, col_1_index, col_2_index,
                   k, membership, output_path, std_df, name, title):
    # Plot the clusters
    for cluster in range(0, k):
        color = cluster_colors[cluster]
        plt.plot(std_df.iloc[membership[cluster]][std_df.columns[col_1_index]],
                 std_df.iloc[membership[cluster]][std_df.columns[col_2_index]], color + 'x')

    # Plot the means
    for cluster_index in range(len(cluster_centers)):
        cluster_center = cluster_centers[cluster_index]
        color = cluster_colors[cluster_index]
        plt.plot(cluster_center[cluster_center.index[col_1_index]], cluster_center[cluster_center.index[col_2_index]],
                 color + 'o')
    plt.title(title)
    plt.xlabel(std_df.columns[col_1_index])
    plt.ylabel(std_df.columns[col_2_index])
    plt.savefig(os.path.join(output_path, "{0}.png".format(name)))


def zscore(dataframe):
    """
    Standardize a given data frame, exclude the first column (assume it is a label)
    :param dataframe:
    :return:
    """
    result = dataframe.copy()

    # Standardize each column except for the first
    mean = dataframe.mean()
    std = dataframe.std()

    print("Mean:\n", mean.to_numpy())
    print("Std:\n", std.to_numpy())

    # Only update all but first column
    result = (result - mean) / std
    print("Z-score Result:\n", result.to_numpy())
    return result


df = pd.read_csv("diabetes.csv")
kmeans(df, 6, 7, 2, ".", df.columns[6:8])