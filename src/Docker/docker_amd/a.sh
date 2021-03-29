#!/bin/bash/
curl -X PUT http://localhost:9200/person?pretty 
'{
	"mappings": {
		"_doc": {
			"properties": {
				"name": {
					"type": "text"
				},
				"author": {
					"type": "text"
				},
				"email": {
					"type": "string",
					"analyzer": "keyword"
				},
				"feature_face": {
					"type": "dense_vector",
					"dims": 512
				},
				"timestamp": {
					"type": "date",
					"format": "yyyy-MM-dd"
				}
			}
		}

	}

}'
