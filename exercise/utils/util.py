
import json
import numpy as np

from multiprocessing import Process

import socket
from contextlib import closing

from MLEng_Exercise.configs.config_variables import KAFKA_HOST

from exercise.utils import logger

log = logger.get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def check_socket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port))


def get_hosts():
    hostsWithPorts = KAFKA_HOST.split(",")
    for hostWithPort in hostsWithPorts:
        hostAndPort = hostWithPort.split(":")
        host = ""
        port = ""
        if (len(hostAndPort) == 1):
            host = hostAndPort[0]
            port = 9092
        elif (len(hostAndPort) == 2):
            host = hostAndPort[0]
            port = hostAndPort[1]
        else:
            raise ValueError("Invalid host ip and port!!")

    connected = check_socket(host, int(port))
    if (connected != 0):
        log.info("Couldn't able to connect kafka host " + str(host) + ":" + str(port) + "\n")
        raise ValueError("Invalid hosts " + KAFKA_HOST)

    return KAFKA_HOST


def multi_process_tasks(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()