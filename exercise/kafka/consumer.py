
"""

The deserialization process is easy with parquet as it can be read with Parquet and
Apache Arrow. Also, this file format won't change any model parameters.

Similarly, for JSON it is the same. We need to assign the saved parameters to a new instance
of the same class.

Author : Guru Prasad Venkata Raghavan

"""

from confluent_kafka import Consumer
from exercise.utils import logger

log = logger.get_logger(__name__)


class KafkaConsumer:
    """
    Initializes the Kafka Consumer that reads the messages from the broker.
    If there is an error message in rare cases, it will decode the message.
    """
    def __init__(self, hosts):
        """
        It initializes the Kafka Hosts and also the consumer configs.

        hosts: String
        """
        self.hosts = hosts
        self.c = Consumer({
            'bootstrap.servers': hosts,
            'group.id': 'mleng-exercise',
            'default.topic.config': {'auto.offset.reset': 'smallest'}
            })

        self.c.subscribe(['people-list-input-queue'])

    def read(self):
        """
        This function reads the message from the topic.
        """
        try:
            log.info("Reading....... from " + self.hosts)
            message = self.c.poll(timeout=3.0)
            if message:
                if not message.error():
                    message_json = message.value().decode('utf-8')
                    log.info("Message from kafka " + message_json)
                    return eval(str(message_json))
        except Exception:
            log.exception("Error while reading the message:")