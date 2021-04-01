#!/bin/bash/
curl -X PUT http://localhost:9200/mtct_person?pretty -H 'Content-Type: application/json' -d' {
        "mappings": {
                "properties": {
                        "fullname": {
                                "type": "text"
                        },
			"user_id": {
				"type": "integer"
			},
                        "username": {
                                "type": "text"
                        },
                        "email": {
                                "type": "text",
                                "analyzer": "keyword"
                        },
			"role_id": {
				"type": "integer"
			},
                        "author": {
                                "type": "text"
                        },
			"password_hash": {
				"type": "text"
			},	
                        "timestamp": {
                                "type": "date",
                                "format": "yyyy-MM-dd"
                        },
			"confirm": {
				"type": "boolean"
			
			}
                
                }

        }

}'

curl -X PUT http://localhost:9200/mtct_arcFace?pretty -H 'Content-Type: application/json' -d' {
	"mappings": {
		"properties": {
			"user_id": {
				"type": "integer"
			},
			"face_vector": {
				"type": "dense_vector",
				"dims": 2048
			},
			"timestamp": {
				"type": "date",
				"format": "yyyy-MM-dd"
			}
		
		}	
	}

}'

curl -X PUT http://localhost:9200/mtct_retinaface?pretty -H 'Content-Type: application/json' -d' {
        "mappings": {
                "properties": {
			"user_id": {
				"type": "integer"
			},
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
