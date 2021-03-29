curl -X PUT httP://localhost:9200/mtct_person?pretty -H 'Content-Type: application/json' -d
'{
	"mappings": {
		"_doc": {
			"properties": {
				"name": {
					"type": "text"	
				},	
				"sex": {
					"type": "text"
				},
				"datatime": {
					"type": "date"
				}
			}	
		}
		
	}
}
';
