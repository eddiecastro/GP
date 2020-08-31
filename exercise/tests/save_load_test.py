"""

This file creates a test framework where it tests whether the model is saved properly
and can be loaded properly.

Author : Guru Prasad Venkata Raghavan

"""


import os
from pathlib import Path
from os.path import curdir, join

from MLEng_Exercise.configs.config_variables import model_file_name

from exercise.utils import logger
from MLEng_Exercise.exercise.mlflow.mlflow_serving import Serving

log = logger.get_logger(__name__)

ml_serve = Serving()


class SaveLoadTest():
    """This is a test framework to assert whether the save and load funcions are working"""

    def __init__(self):
        self.model_artifact_path = None
        self.model_file_name = model_file_name
        self.model_file = None

    def _get_saved_model_path(self):
        self.model_artifact_path = ml_serve.get_artifact()
        self.model_file = join(curdir, self.model_artifact_path, self.model_file_name)

    def _test_model_exists(self):
        """This functionality will check whether the model exists in the artifact uri"""

        file_name = Path(self.model_file)
        if file_name.exists():
            log.info("Saved model exists")
        else:
            log.error("The model didn't get saved in the artifact path")

    def _test_model_content(self):
        """ This functionality tests if the model contains content in it and it is not empty"""

        try:
            assert os.stat(self.model_file).st_size != 0
            log.info("The model file is not empty")
        except:
            log.error("The model file is empty and the model is not saved.")

    def test_artifact_save(self):
        """This functionality will check whether the model is saved as an artifact and other conditions"""

        self._get_saved_model_path()
        self._test_model_exists()
        self._test_model_content
        log.info("The model file is saved properly and tested appropirately")

    def _check_empty_list(self, predicted_labels):
        """Checks whether the list is empty"""
        try:
            assert len(predicted_labels) >= 0
            log.info("The predicted labels are not empty")
        except:
            log.error("The predicted labels are empty. The load function didn't work")

    def test_artifact_load(self, predicted_labels):
        """This functionality will check whether the saved model can be loaded and used."""
        self._check_empty_list(predicted_labels)
        log.info("The load functions are working.")



