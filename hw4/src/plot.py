import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_pca_scatterplot(title, y, pcas):
    '''
    Plots the computed principal components analysis data
    leveraging pandas dataframe for easy data manipulation.

    :param y: The targets data
    :param pcas: The calculated principal components of the data

    :return none
    '''
    pcas_df = pd.DataFrame(pcas, columns=['PC1', 'PC2'])
    pcas_df = pd.concat(
        [pcas_df, pd.DataFrame(y, columns=['ID'])], axis=1)

    plt.figure(figsize=(10, 8))
    plt.title(title, "- Labeled Faces in the Wild")

    sp = sns.scatterplot(x='PC1', y='PC2', hue='ID', s=100,
                         data=pcas_df, palette='colorblind', legend='auto')
    handles, labels = sp.get_legend_handles_labels()
    sp.legend(title='ID', handles=handles, labels=labels, ncol=2,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
