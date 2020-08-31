
import pandas as pd
import numpy as np


class Preprocess:
    """

    """

    def __init__(self, file_path, col_x, col_y):
        """

        """
        self.file_path = file_path
        self.col_x = col_x
        self.col_y = col_y
        self.df = None
        self.train = None
        self.test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def read_csv(self):
        """

        :param file_path:
        :return:
        """
        self.df = pd.read_csv(self.file_path)

    def clean_dataframe(self):
        """

        :return:
        """
        assert isinstance(self.df, pd.DataFrame), "The input is not a dataframe"
        self.df.dropna(inplace=True)
        clean_indexes = ~self.df.isin([np.nan, np.inf, -np.inf]).any(1)
        return self.df[clean_indexes]

    def train_test_split(self, train_percent=.7):
        """

        :param train_percent:
        :param seed:
        :return:
        """
        self.train, self.test = np.split(self.df.sample(frac=1), [int(train_percent * len(self.df))])

    def get_x_y_split(self):
        """

        :return:
        """
        self.X_train = self.train[self.col_x].to_numpy()
        self.y_train = self.train[self.col_y].to_numpy()
        self.X_test = self.test[self.col_x].to_numpy()
        self.y_test = self.test[self.col_y].to_numpy()

    def run_all(self):
        """

        :return:
        """
        self.read_csv()
        self.clean_dataframe()
        self.train_test_split()
        self.get_x_y_split()
        return self.X_train, self.y_train, self.X_test, self.y_test