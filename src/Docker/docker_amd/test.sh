#!/bin/bash/
curl -X PUT "localhost:9200/customer/_doc/3?pretty" -H 'Content-Type: application/json' -d' { 
	"name": "Testname",
	"sex": "Nam",
	"email": "quanbn1336@gmail.com"

}'
