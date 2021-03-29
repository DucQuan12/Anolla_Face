import os
import sys
import elasticsearch as Elasticsearch
import asyncio
import base64
import logging

logging.basicConfig()
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])


class Search(object):
    def __init__(self, feature):
        self.feature_base64 = feature

    def _index(self, feature):
        pass
    def search_feature(self, feature_base64, object_image, _search=True):
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
                query = es.search(index="person", query=query)
                query.sorted()
                query_list.append(query[0])
            else:
                logging.info("No Search")
            return query_list

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
            query = await es.search(index='person', query=query)
            query.sorted()
            list_query.append(query[0])
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete()

# async wait
