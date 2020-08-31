
"""

The deserialization process is easy with parquet as it can be read with Parquet and
Apache Arrow. Also, this file format won't change any model parameters.

Similarly, for JSON it is the same. We need to assign the saved parameters to a new instance
of the same class.

Author : Guru Prasad Venkata Raghavan

"""

from confluent_kafka import Producer
import json
from exercise.utils import logger

log = logger.get_logger(__name__)


class KafkaProducer:
    """
    Initializes the Kafka Producer that publishes the messages to the broker.
    """
    def __init__(self, hosts):
        """
        It initializes the Kafka Hosts and also the producer configs.

        :param hosts:
        """
        self.hosts = hosts
        self.producer = Producer(
            {'bootstrap.servers': hosts, 'api.version.request': False}
        )

    def send(self, topic, message):
        """
        This function produces the message to the topic

        topic: String
        message: String
        """
        log.info("sending to hosts" + self.hosts)
        log.info("Sending message..... " + str(message))
        self.producer.produce(topic, json.dumps(message).encode('utf-8'))
        self.producer.flush()
