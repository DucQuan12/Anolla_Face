#!/bin/bash/
curl -X PUT http://localhost:9200/detect_person?pretty -H 'Content-Type: application/json' -d' {
        "mappings": {  
		"properties": {
			"face_feature": {
				"type": "dense_vector",
				"dims": 2048
			},
			"user_name": {
				"type": "text"
			},
			"timestamp": {
				"type": "date",
				"format": "yyyy-MM-dd"
			}
		}
	}
}'
curl -X PUT http://localhost:9200/detect_face?pretty -H 'Content-Type: application/json' -d' {
        "mappings": {
		"properties": {
			"path_image": {
				"type": "text"	
			},
			"timestamp": {
				"type": "date",
				"format": "yyyy-MM-dd"
			}
                }
        }

}'
