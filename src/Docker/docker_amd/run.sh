#!/bin/bash
sudo sysctl -w vm.max_map_count=262144
sudo docker run -p 9200:9200 -p 9300:9300 --name elasticsearch -e "discovery.type=single-node" elasticsearch:7.9.0
