
"""

The deserialization process is easy with parquet as it can be read with Parquet and
Apache Arrow. Also, this file format won't change any model parameters.

Similarly, for JSON it is the same. We need to assign the saved parameters to a new instance
of the same class.

Author : Guru Prasad Venkata Raghavan

"""

import json

import numpy as np
import pandas as pd


class DeSerialization:
    """
    """

    def __init__(self, model_file_name, X_test, clf):
        """
        This functionality initilizes the deserialization parameters.
        """
        self.filename = model_file_name
        self.X_test = X_test
        self.clf = clf

    def _load_attributes_for_json(self, data):
        """
        This functionality assigns the saved parameters to the model.

        :param data: The input data would be the model parameters saved
        :return:
        """
        self.clf.coef_ = np.array(data['coef'])
        self.clf.intercept_ = np.array(data['intercept'])
        self.clf.classes_ = np.array(data['classes'])
        self.clf.n_iter_ = np.array(data['n_iter'])
        self.clf.penalty = np.array(data['penalty'])
        self.clf.loss = np.array(data['loss'])
        self.clf.dual = np.array(data['dual'])
        self.clf.tol = np.array(data['tol'])
        self.clf.C = np.array(data['C'])
        self.clf.multi_class = np.array(data['multi_class'])
        self.clf.fit_intercept = np.array(data['fit_intercept'])
        self.clf.intercept_scaling = np.array(data['intercept_scaling'])
        self.clf.verbose = np.array(data['verbose'])
        self.clf.max_iter = np.array(data['max_iter'])
        predicted_labels = self.clf.predict(self.X_test)
        return predicted_labels

    def _load_attributes_for_parquet(self, data):
        """
        This functionality assigns the saved parameters to the model from parquet file.

        :param data: The input data would be the model parameters saved
        :return:
        """
        self.clf.coef_ = np.array(data["coef"]).reshape((data["shape"][0], data["shape"][1]))
        self.clf.intercept_ = np.array(data["intercept"])[0:data["shape"][0]]
        self.clf.classes_ = np.array(data["classes"])[0:data["shape"][0]]
        self.clf.n_iter_ = np.array(data["n_iter"])[0]
        self.clf.penalty = np.array(data["penalty"])[0]
        self.clf.loss = np.array(data["loss"])[0]
        self.clf.dual = np.array(data["dual"])[0]
        self.clf.tol = np.array(data["tol"])[0]
        self.clf.C = np.array(data["C"])[0]
        self.clf.multi_class = np.array(data["multi_class"])[0]
        self.clf.fit_intercept = np.array(data["fit_intercept"])[0]
        self.clf.intercept_scaling = np.array(data["intercept_scaling"])[0]
        self.clf.verbose = np.array(data["verbose"])[0]
        self.clf.max_iter = np.array(data["max_iter"])[0]
        predicted_labels = self.clf.predict(self.X_test)
        return predicted_labels

    def _read_model_from_json(self):
        """
        """
        with open(self.filename, 'r') as file:
            data = json.load(file)
        predicted_labels = self._load_attributes_for_json(data)
        return predicted_labels

    def _read_model_from_parquet(self):
        """
        """
        data = pd.read_parquet(self.filename, engine='pyarrow')
        predicted_labels = self._load_attributes_for_parquet(data)
        return predicted_labels

    def deserialize_model(self):
        if ".json" in self.filename:
            predicted_labels = self._read_model_from_json()
        elif ".parquet" in self.filename:
            predicted_labels = self._read_model_from_parquet()
        return predicted_labels
