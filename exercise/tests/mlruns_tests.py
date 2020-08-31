
"""

This file evaluates the metrics scores of both the persisted model and hydrated model are the same.
The respective scores of both persisted model and hydrated models for each runs are tracked in MLFlow.
This test framework will assert if they are same or raise an error.

Author : Guru Prasad Venkata Raghavan

"""

from exercise.utils import logger
from MLEng_Exercise.exercise.utils.util import multi_process_tasks

log = logger.get_logger(__name__)


class TestMLRuns:
    """This is a test framework to find whether the hydrated model run and the persisted model runs are the same."""

    def __init__(self, hydrated_model_score, persisted_model_score):
        self.hydrated_model_score = hydrated_model_score
        self.persisted_model_score = persisted_model_score

    def test_recall(self):
        """This functionality finds whether the recall scores are the same."""
        hydrated_model_recall = self.hydrated_model_score["metrics.Recall Score"]
        persisted_model_recall = self.persisted_model_score["metrics.Recall Score"]
        try:
            assert hydrated_model_recall == persisted_model_recall
            log.error("The recall scores for the hydrated model and persisted model are the same")
        except:
            log.error("The recall scores for the hydrated model and Persisted model are not the same.")


    def test_accuracy(self):
        """This functionality finds the accuracy scores are the same."""
        hydrated_model_accuracy = self.hydrated_model_score["metrics.accuracy"]
        persisted_model_accuracy = self.persisted_model_score["metrics.accuracy"]
        try:
            assert hydrated_model_accuracy == persisted_model_accuracy
            log.error("The accuracy scores for the hydrated model and persisted model are the same")
        except:
            log.error("The accuracy scores for the hydrated model and Persisted model are not the same.")

    def test_f1(self):
        """This functionality find whether the F1 scores are the same."""
        hydrated_model_f1 = self.hydrated_model_score["metrics.F1 Score"]
        persisted_model_f1 = self.persisted_model_score["metrics.F1 Score"]
        try:
            assert hydrated_model_f1 == persisted_model_f1
            log.error("The F1 scores for the hydrated model and persisted model are the same")
        except:
            log.error("The f1 scores for the hydrated model and Persisted model are not the same.")

    def test_precision(self):
        """This functionality find whether the Precision scores are the same."""
        hydrated_model_precision = self.hydrated_model_score["metrics.Precision Score"]
        persisted_model_precision = self.persisted_model_score["metrics.Precision Score"]
        try:
            assert hydrated_model_precision == persisted_model_precision
            log.error("The Precision scores for the hydrated model and persisted model are the same")
        except:
            log.error("The Precision scores for the hydrated model and Persisted model are not the same.")

    def runall(self):
        """This functionality runs all the evaluation metrics in parallel."""
        multi_process_tasks(self.test_f1, self.test_accuracy, self.test_recall, self.test_precision)
        log.info("Parallel run of the mlruns_test is completed.")



