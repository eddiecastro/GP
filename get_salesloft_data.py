
import uuid
import json
import requests
from time import sleep
from datetime import datetime

from exercise.utils import logger
from MLEng_Exercise.exercise.mongo.mongodb import MongoDB
from MLEng_Exercise.exercise.kafka.producer import KafkaProducer

from MLEng_Exercise.exercise.utils.util import get_hosts

log = logger.get_logger(__name__)

def get_salesloft_data(url, authorization):
    """

    """
    payload = {}
    headers = {'Authorization': authorization}
    response = requests.request("GET", url, headers=headers, data=payload)

    if response.status_code != 200:
        print('Failed to get data:', response.status_code)
    else:
        data = json.loads(response.text)

        log.info('[*] Pushing Raw SalesLoft Data to MongoDB ')
        mongo_db = MongoDB(database_name='MLEng_Exercise_DB', collection_name='salesloft_data')

        log.info('[!] Inserting raw SalesLoft data')
        mongo_db.insert({"Raw SalesLoft Data":data,"Version":str(dateTime)})

        return data["data"]


if __name__ == "__main__":
    hosts = get_hosts()
    producer = KafkaProducer(hosts)

    url = "https://api.salesloft.com/v2/people.json?API_KEY=384325a37c7096723de4210e349383afcfecc623e342f7a10650a95f13b3a0b0"
    authentication = "Bearer v2_ak_10822_384325a37c7096723de4210e349383afcfecc623e342f7a10650a95f13b3a0b0"

    while True:
        try:
            dateTime = datetime.now()
            data = get_salesloft_data(url, authentication)
            job_titles = []
            for value in data:
                if 'title' in value:
                    job_titles.append({"Job Id":value["id"],"Job Title":value["title"],"Version":str(dateTime)})

            mongo_db = MongoDB(database_name='MLEng_Exercise_DB', collection_name='predict_input_data')

            for collection in job_titles:
                print('[!] Inserting - ', collection)
                mongo_db.insert(collection)

            log.info('[*] Producing mongo collection details for predicting data ')

            message = {"database_name": "MLEng_Exercise_DB", "collection_name": "predict_input_data", \
                       "message_id": str(uuid.uuid4()), "version": str(dateTime)}

            producer.send('people-list-input-queue', message)

        except Exception as error:
            log.exception('Error :')
        sleep(10)
