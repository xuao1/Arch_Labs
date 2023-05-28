#!/bin/bash

echo "N = 512"
echo "9" | nvprof ./GPU1
echo -e "\n"
echo "N = 1024"
echo "10" | nvprof ./GPU1
echo -e "\n"
echo "N = 2048"
echo "11" | nvprof ./GPU1
echo -e "\n"
echo "N = 4096"
echo "12" | nvprof ./GPU1
echo -e "\n"