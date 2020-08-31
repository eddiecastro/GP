
"""

This file evaluates the results of the machine learning for efficient performance of the model.
Since, it is a multi-class classification model, the following evaluation metrics
of F1, Accuracy, Recall and Precision are used.

Author : Guru Prasad Venkata Raghavan

"""

from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score

from MLEng_Exercise.exercise.mlflow.mlflow_tracking import Tracking
from MLEng_Exercise.exercise.utils.util import multi_process_tasks

tracking = Tracking()


class ClassificationTests:
    """This is a test framework to find whether the results of the model satisfy the evaluation metrics."""

    def __init__(self, actual_y, pred_y):
        self.actual_y = actual_y
        self.pred_y = pred_y
        self.f1 = None
        self.accuracy = None
        self.recall = None
        self.precision = None

    def test_f1(self):
        """The F1 score can be interpreted as a weighted average of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0. -- SkLearn"""
        self.f1 = f1_score(self.actual_y, self.pred_y, average='micro')
        tracking.log_metrics("F1 Score", self.f1)

    def test_accuracy(self):
        """
        The set of labels predicted for a sample must exactly match the corresponding
        set of labels in y_true.      --SkLearn
        """
        self.accuracy = accuracy_score(self.actual_y, self.pred_y)
        tracking.log_metrics("accuracy", self.accuracy)

    def test_recall(self):
        """
        The recall is the ratio tp / (tp + fn) where tp is the number of true positives
        and fn the number of false negatives. The recall is intuitively the ability of
        the classifier to find all the positive samples.  --SkLearn
        """
        self.recall = recall_score(self.actual_y, self.pred_y, average='micro')
        tracking.log_metrics("Recall Score", self.recall)

    def test_precision(self):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives
        and fp the number of false positives. The precision is intuitively the ability of
        the classifier not to label as positive a sample that is negative. The best value
        is 1 and the worst value is 0.                   --SkLearn
        """
        self.precision = precision_score(self.actual_y, self.pred_y, average='micro')
        tracking.log_metrics("Precision Score", self.precision)

    def runall(self):
        """
        This functionality runs all the metrics in parallel.
        """
        multi_process_tasks(self.test_f1, self.test_accuracy, self.test_recall, self.test_precision)
        log.info("Parallel runs of the classification_tests are completed.")




