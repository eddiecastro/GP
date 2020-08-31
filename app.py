from time import sleep
from os.path import join, curdir, abspath

from MLEng_Exercise.exercise.utils.util import get_hosts
from MLEng_Exercise.configs.config_variables import(
    PATH, input_file, col_x, col_y, model_file_path)

from exercise.utils import logger
from MLEng_Exercise.exercise.model import SeniorityModel
from MLEng_Exercise.exercise.kafka.consumer import KafkaConsumer
from MLEng_Exercise.exercise.kafka.producer import KafkaProducer
from MLEng_Exercise.exercise.mlflow.mlflow_tracking import Tracking
from MLEng_Exercise.exercise.mlflow.mlflow_serving import Serving
from MLEng_Exercise.exercise.preprocess.preprocess_data import Preprocess

log = logger.get_logger(__name__)



data_filepath = abspath(join(curdir, PATH, input_file))
pre_process = Preprocess(data_filepath, col_x, col_y)

def pub_sub():
    hosts = get_hosts()
    log.info(hosts)
    consumer = KafkaConsumer(hosts)
    producer = KafkaProducer(hosts)
    return consumer, producer


if __name__ == "__main__":
    consumer, producer = pub_sub()

    tracking = Tracking()
    tracking._set_mlflow_configs()

    serving = Serving()

    X_train, y_train, X_test, y_test = pre_process.run_all()

    seniorty_model = SeniorityModel()
    seniorty_model.fit(X_train, y_train)
    seniorty_model.predict(X_test, y_test, "Predict Method", tracking)
    seniorty_model.save(model_file_path, "Save Model",tracking)
    seniorty_model.load(model_file_path, X_test, y_test, "Persisted vs Hydrated Model", tracking)
    seniorty_model.test_hydrated_vs_persisted(serving, "Predict Method","Persisted vs Hydrated Model")

    while True:
        try:
            # log.info("Initial Memory Used : " + str(getrusage(RUSAGE_SELF).ru_maxrss))
            log.info('Consume the message')
            message = consumer.read()
            log.info('Completed consuming the message')

            if message is not None:
                log.info(message)
                print(seniorty_model.online_predict(message))
        except Exception as error:
            log.exception('Error :')
        sleep(10)




