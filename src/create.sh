#!/bin/bash/
mkdir key
openssl req -newkey rsa:2048 -nodes -keyout ./key/server.key -x509 -days 365 -out ./key/server.crt
