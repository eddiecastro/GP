
"""

The deserialization process is easy with parquet as it can be read with Parquet and
Apache Arrow. Also, this file format won't change any model parameters.

Similarly, for JSON it is the same. We need to assign the saved parameters to a new instance
of the same class.

Author : Guru Prasad Venkata Raghavan

"""

import mlflow
from mlflow.tracking import MlflowClient

from MLEng_Exercise.configs.config_variables import uri, experiment

client = MlflowClient()


class Tracking:

    def __init__(self):
        self.tracking_uri = uri
        self.experiment_name = experiment
        self.exp_id = None

    def _set_mlflow_configs(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return

    def _get_experiment_id(self):
        self.exp_id = client.get_experiment_by_name(self.experiment_name).experiment_id
        return

    def start_runs(self,run_name):
        mlflow_run = mlflow.start_run(experiment_id=self.exp_id,run_name=run_name)
        return mlflow_run

    def end_runs(self):
        mlflow.end_run()
        return

    def log_metrics(self, metric_name, metric):
        mlflow.log_metric(metric_name, metric)
        return

    def log_params(self, metric_name, metric):
        mlflow.log_metric(metric_name, metric)
        return

    def log_artifacts(self,artifact_path):
        mlflow.log_artifact(artifact_path)
        return

