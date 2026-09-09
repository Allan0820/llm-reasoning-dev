#!/usr/bin/env bash
g++ -Wall -Werror -std=gnu++17 CppRDFoxDemo.cpp -o CppRDFoxStaticDemo -I../../include -Wl,--whole-archive ../../lib/libRDFox-static.a -Wl,--no-whole-archive -ldl -lpthread
cp ../data/* .
./CppRDFoxStaticDemo
        