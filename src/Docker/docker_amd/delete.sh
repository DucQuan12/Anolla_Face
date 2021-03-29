#!/bin/bash
curl -X DELETE http://localhost:9200/mtct_person?pretty
curl -X DELETE http://localhost:9200/mtct_retinaface?pretty
curl -X DELETE http://localhost:9200/mtct_arcface?pretty
