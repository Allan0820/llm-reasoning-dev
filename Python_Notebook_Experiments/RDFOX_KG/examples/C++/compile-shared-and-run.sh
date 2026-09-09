#!/usr/bin/env bash
g++ -Wall -Werror -std=gnu++17 CppRDFoxDemo.cpp -o CppRDFoxSharedDemo -I../../include -lRDFox -L../../lib
cp ../data/* .
LD_LIBRARY_PATH=../../lib DYLD_FALLBACK_LIBRARY_PATH=../../lib ./CppRDFoxSharedDemo
        