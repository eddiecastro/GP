
"""

The deserialization process is easy with parquet as it can be read with Parquet and
Apache Arrow. Also, this file format won't change any model parameters.

Similarly, for JSON it is the same. We need to assign the saved parameters to a new instance
of the same class.

Author : Guru Prasad Venkata Raghavan

"""

import mlflow

from MLEng_Exercise.exercise.mlflow.mlflow_tracking import Tracking


class Serving(Tracking):

    def __init__(self):
        super().__init__()
        self.artifact = None

    def get_artifact(self):
        artifact_path = mlflow.get_artifact_uri()
        return artifact_path

    def search_runs(self):
        runs = mlflow.search_runs(self.exp_id)
        return runs