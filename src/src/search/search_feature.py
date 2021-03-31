import elasticsearch as Elasticsearch
import asyncio
import logging
import base64
import time
import sys
import os

logging.basicConfig()


class Search(object):
    def __init__(self, host, port):
        es = self.Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        host = self.host
        port = self.port
        list_index = []

    def _index(self, name_index):
        for index in self.es.indices.get('*'):
            self.list_index.append(index)
        return self.list_index

    def _update(self, query_update, embedded_feature):
        user_id = query_update.get('user_id')
        face_vector = query_update.get('face_vector')
        query = {
            "mappings": {
                "doc": {
                    "properties": {
                        "user_id": user_id,
                        "face_vector": face_vector,
                        "timestamp": int(time.time())
                    }
                }
            }
        }

    def _search_feature(self, feature_base64, object_image, min_score=0.7, _search=True):
        query_list = []
        for i in object_image:
            query = {
                "size": 5,
                "query": {
                    "match_all": {
                        "feature": feature_base64
                    }
                },
                "script": {
                    "source": "1/(1 + l2nor)",
                    "params": {
                        "queryVector": list(),
                    }
                }
            }
            if _search is True:
                query = self.es.search(index="person", query=query)
                query.sorted()
                query_list.append(query[0])
            else:
                logging.info("No Search")
            return query_list[:5]

    async def search_query(self, feature_base64, object_image, _search=True):
        list_query = []
        for i in object_image:
            query = {
                "size": 5,
                "query": {
                    "match_all": {
                        "feature": feature_base64
                    },
                    "script": {
                        "source": "1/(1+lnNorm)",
                        "params": {
                            "queryVector": list(),
                        }
                    }
                }
            }
        if _search is True:
            query = await self.es.search(index='person', query=query)
            query.sorted()
            list_query.append(query[0])
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete()

    def __str__(self):
        return self.__class__.__name__

# async wait
