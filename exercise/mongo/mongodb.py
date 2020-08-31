from pymongo import MongoClient


class MongoDB(object):
    def __init__(self, host='localhost', port=27017, database_name=None, collection_name=None):
        try:
            self._connection = MongoClient(host=host, port=port, maxPoolSize=200)
        except Exception as error:
            raise Exception(error)
        self._database = None
        self._collection = None
        if database_name:
            self._database = self._connection[database_name]
        if collection_name:
            self._collection = self._database[collection_name]

    def insert(self, post):
        # add/append/new single record
        post_id = self._collection.insert_one(post).inserted_id
        return post_id

    def retrieve_all(self):
        # Fetch all data
        collection_data = self._collection.find()
        return collection_data

    def retrieve_last_n_records(self, N):
        # Fetch last N records
        collection_data = self._collection.find().skip(self._collection.count() - N)
        return collection_data

    def retrieve_by_condition(self, filter):
        collection_data = self._collection.find(filter)
        return collection_data

    def insert_all(self, data):
        for collection in data:
            self.insert(collection)
