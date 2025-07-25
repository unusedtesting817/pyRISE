import matplotlib.pyplot as plt
import pandas as pd

def generate_table(data, title):
    """
    Generate a table from a pandas DataFrame.

    Args:
        data (pd.DataFrame): The data to be displayed in the table.
        title (str): The title of the table.

    Returns:
        The table.
    """
    print(title)
    print(data)
    return data

def generate_plot(data, title):
    """
    Generate a plot from a pandas DataFrame.

    Args:
        data (pd.DataFrame): The data to be plotted.
        title (str): The title of the plot.
    """
    data.plot()
    plt.title(title)
    plt.show()
