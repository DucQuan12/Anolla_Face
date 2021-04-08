#!/bin/bash/
python3 -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/server.proto
mkdir key
openssl req -newkey rsa:2048 -nodes -keyout ./key/server.key -x509 -days 365 -out ./key/server.crt