import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from exercise.utils import logger

from MLEng_Exercise.exercise.mongo.mongodb import MongoDB

from MLEng_Exercise.exercise.tests.classification_tests import ClassificationTests
from MLEng_Exercise.exercise.tests.save_load_test import SaveLoadTest
from MLEng_Exercise.exercise.tests.mlruns_tests import TestMLRuns

from MLEng_Exercise.exercise.serialization.serialize_models import Serialization
from MLEng_Exercise.exercise.serialization.deserialize_models import DeSerialization

log = logger.get_logger(__name__)


def clean_transform_title(job_title):
    """Clean and transform job title. Remove punctuations, special characters,
    multiple spaces etc.
    """
    if not isinstance(job_title, str):
        return ''
    new_job_title = job_title.lower()
    special_characters = re.compile('[^ a-zA-Z]')
    new_job_title = re.sub(special_characters, ' ', new_job_title)
    extra_spaces = re.compile(r'\s+')
    new_job_title = re.sub(extra_spaces, ' ', new_job_title)
    
    return new_job_title


class SeniorityModel:
    """Job seniority model class. Contains attributes to fit, predict,
    save and load the job seniority model.
    """
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.shape = None
    
    def _check_for_array(self, variable):
        if not isinstance(variable, (list, tuple, np.ndarray)):
            raise TypeError("variable should be of type list or numpy array.")
        return
    
    def _data_check(self, job_titles, job_seniorities):
        self._check_for_array(job_titles)
        self._check_for_array(job_seniorities)
        
        if len(job_titles) != len(job_seniorities):
            raise IndexError("job_titles and job_seniorities must be of the same length.")
        
        return
        
    def fit(self, job_titles, job_seniorities):
        """Fits the model to predict job seniority from job titles.
        Note that job_titles and job_seniorities must be of the same length.
        
        Parameters
        ----------
        job_titles: numpy array or list of strings representing job titles
        job_seniorities: numpy array or list of strings representing job seniorities
        """
        self._data_check(job_titles, job_seniorities)
        
        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        
        self.vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
        vectorized_data = self.vectorizer.fit_transform(cleaned_job_titles)
        self.model = LinearSVC()
        self.model.fit(vectorized_data, job_seniorities)
        
        return

    def _vectorize_data(self, job_titles):
        """

        Parameters
        ----------
        :param job_titles:
        :return:
        """

        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        vectorized_data = self.vectorizer.transform(cleaned_job_titles)
        return vectorized_data

    def _push_to_mongo(self, dbname, collection, job_seniority_predictions ):
        mongo_db = MongoDB(database_name=dbname, collection_name=collection)
        mongo_db.insert_all(job_seniority_predictions)
        return

    def predict(self, job_titles, job_seniorities,run_name, tracking):
        """
        """
        self._data_check(job_titles, job_seniorities)

        vectorized_job_titles = self._vectorize_data(job_titles)
        vectorized_job_seniorities = self._vectorize_data(job_seniorities)
        predicted_labels = self.model.predict(vectorized_job_titles)
        vectorized_predicted_labels = self._vectorize_data(predicted_labels)

        tracking.start_runs(run_name)

        test = ClassificationTests(vectorized_job_seniorities,vectorized_predicted_labels)
        test.runall()

        tracking.end_runs()

        log.info('[*] Pushing predicted labels to Mongo DB ')

        # Insert Job Seniority predictions into mongo
        job_seniority_predictions = []
        for label_num in range(0, len(predicted_labels)):
            job_seniority_predictions.append({"Job Seniority": predicted_labels[label_num]})

        self._push_to_mongo("MLEng_Exercise_DB", "predicted_labels", job_seniority_predictions)

        return predicted_labels


    def online_predict(self, kafka_message):
        """
        """

        # Get the streaming input data from MongoDB
        mongo_db = MongoDB(database_name=kafka_message["database_name"], collection_name=kafka_message["collection_name"])
        log.info('[*] Retrieving data to be predicted from MongoDB ')
        data = list(mongo_db.retrieve_by_condition({"Version": kafka_message["version"]}))

        # Get the Job Seniority predictions
        job_titles = [value["Job Title"] for value in data if "Job Title" in value]
        self._check_for_array(job_titles)
        vectorized_job_titles = self._vectorize_data(job_titles)
        predicted_labels = self.model.predict(vectorized_job_titles)

        log.info('[*] Pushing Online predictions to Mongo DB ')

        # Insert Job Seniority predictions into mongo
        job_seniority_predictions = []
        for label_num in range(0, len(predicted_labels)):
            job_seniority_predictions.append({"Job Id": data[label_num]["Job Id"],
                                              "Job Seniority": predicted_labels[label_num],
                                              "Version": data[label_num]["Version"]})

        self._push_to_mongo("MLEng_Exercise_DB", "online_predicted_labels", job_seniority_predictions)

        return

    def save(self,filename,run_name, tracking):
        """
        """
        tracking.start_runs(run_name)
        serialization = Serialization(self.model, filename, tracking)
        serialization.serialize_model()

        save_check = SaveLoadTest()
        save_check.test_artifact_save()
        tracking.end_runs()

        return

    def load(self,filename,job_titles,job_seniorities,run_name, tracking):
        clf = LinearSVC()
        vectorized_job_titles = self._vectorize_data(job_titles)
        deserialize = DeSerialization(filename, vectorized_job_titles, clf)
        predicted_labels = deserialize.deserialize_model()

        load_check = SaveLoadTest()
        load_check.test_artifact_load(predicted_labels)

        vectorized_job_seniorities = self._vectorize_data(job_seniorities)
        vectorized_predicted_labels = self._vectorize_data(predicted_labels)

        tracking.start_runs(run_name)
        test = ClassificationTests(vectorized_job_seniorities, vectorized_predicted_labels)
        test.runall()

        tracking.end_runs()

        # Insert Job Seniority predictions into mongo

        log.info('[*] Pushing loaded model predictions to Mongo DB ')

        job_seniority_predictions = []
        for label_num in range(0, len(predicted_labels)):
            job_seniority_predictions.append({"Job Seniority": predicted_labels[label_num]})

        self._push_to_mongo("MLEng_Exercise_DB", "loaded_model_predictions", job_seniority_predictions)

        return

    def test_hydrated_vs_persisted(self, serving, hydrated_model_run, persisted_model_run):

        mlruns_df = serving.search_runs()
        hydrated_model_scores = mlruns_df.loc[mlruns_df['tags.mlflow.runName'] == hydrated_model_run].iloc[0].to_dict()
        persisted_model_scores = mlruns_df.loc[mlruns_df['tags.mlflow.runName'] == persisted_model_run].iloc[0].to_dict()
        mlruns_test = TestMLRuns(hydrated_model_scores, persisted_model_scores)
        mlruns_test.runall()

