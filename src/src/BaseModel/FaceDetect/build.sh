#!/bin/bash/:wq
git clone https://github.com/clancylian/retinaface.git
cd ./retinaface
mkdir build
cd build
cmake ../
make
