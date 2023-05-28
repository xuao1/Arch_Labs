#!/bin/bash

echo "N = 512"
echo "BlockSize = 16"
echo "9 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "9 32" | nvprof ./GPU0
echo "BlockSize = 64"
echo "9 64" | nvprof ./GPU0
echo "BlockSize = 128"
echo "9 128" | nvprof ./GPU0
echo "\n"

echo "N = 1024"
echo "BlockSize = 16"
echo "10 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "10 32" | nvprof ./GPU0
echo "BlockSize = 64"
echo "10 64" | nvprof ./GPU0
echo "BlockSize = 128"
echo "10 128" | nvprof ./GPU0
echo "\n"

echo "N = 2048"
echo "BlockSize = 16"
echo "11 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "11 32" | nvprof ./GPU0
echo "BlockSize = 64"
echo "11 64" | nvprof ./GPU0
echo "BlockSize = 128"
echo "11 128" | nvprof ./GPU0
echo "\n"

echo "N = 4096"
echo "BlockSize = 16"
echo "12 16" | nvprof ./GPU0
echo "BlockSize = 32"
echo "12 32" | nvprof ./GPU0
echo "BlockSize = 64"
echo "12 64" | nvprof ./GPU0
echo "BlockSize = 128"
echo "12 128" | nvprof ./GPU0
echo "\n"