#!/usr/bin/env bash
gcc -std=c17 -Wall -Werror CRDFoxDemo.c -o CRDFoxSharedDemo -I../../include -lRDFox -L../../lib
cp ../data/* .
LD_LIBRARY_PATH=../../lib DYLD_FALLBACK_LIBRARY_PATH=../../lib ./CRDFoxSharedDemo
        