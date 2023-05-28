#!/bin/bash

echo "N = 512"
echo "BlockSize = 4"
echo "9 4" | nvprof ./GPU0
echo "BlockSize = 8"
echo "9 8" | nvprof ./GPU0
echo "BlockSize = 16"
echo "9 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "9 32" | nvprof ./GPU0
echo "\n"

echo "N = 1024"
echo "BlockSize = 4"
echo "10 4" | nvprof ./GPU0
echo "BlockSize = 8"
echo "10 8" | nvprof ./GPU0
echo "BlockSize = 16"
echo "10 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "10 32" | nvprof ./GPU0
echo "\n"

echo "N = 2048"
echo "BlockSize = 4"
echo "11 4" | nvprof ./GPU0
echo "BlockSize = 8"
echo "11 8" | nvprof ./GPU0
echo "BlockSize = 16"
echo "11 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "11 32" | nvprof ./GPU0
echo "\n"

echo "N = 4096"
echo "BlockSize = 4"
echo "12 4" | nvprof ./GPU0
echo "BlockSize = 8"
echo "12 8" | nvprof ./GPU0
echo "BlockSize = 16"
echo "12 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "12 32" | nvprof ./GPU0
echo "\n"