import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_pca_scatterplot(y, projections):
    '''
    Plots the computed principal components analysis data
    leveraging pandas dataframe for easy data manipulation.

    :param y: The targets data
    :param projections: The calculated principal components 
    of the data

    :return none
    '''
    principal_df = pd.DataFrame(projections, columns=['PC1', 'PC2'])
    principal_df = pd.concat(
        [principal_df, pd.DataFrame(y, columns=['id'])], axis=1)

    plt.figure(figsize=(10, 8))
    plt.title("Principal Component Analysis - Labeled Faces in the Wild")

    # TODO: Need to make the legend a two column
    sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='id', s=100, palette="colorblind")
    plt.show()