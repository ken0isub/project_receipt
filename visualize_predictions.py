import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def model_summary():
    for model_summary in os.listdir('models/summary'):
        df = pd.read_csv('models/summary/' + model_summary, index_col=0)
        print(df)

        df_dummy = pd.get_dummies(df, columns=['2'])
        df_dummy_g = df_dummy.groupby(['1']).sum()
        print(df_dummy_g)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(model_summary[:-4])
        sns.heatmap(df_dummy_g, ax=ax, annot=True, cmap='Reds')
        plt.show()
        plt.close('all')