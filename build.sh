#!/bin/bash

find . -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -style=file -i

mkdir build
cd build

if [ $INSTALL ]; then
    cmake ..; make install
else
    cmake ..; make
fi