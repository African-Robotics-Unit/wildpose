#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize

# Resize (8-bits sample)
./ResizeSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1000.ppm -outputWidth 1000 -info -log resize.log
./ResizeSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.3000.ppm -outputWidth 3000 -info
./ResizeSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1000x1000.ppm -outputWidth 1000 -outputHeight 1000 -info
./ResizeSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1000x1000.padded.ppm -outputWidth 1000 -outputheight 1000 -background 128,128,128 -info


# Resize (12-bits sample)
./ResizeSample  -i 2k_wild.1920x1080.12.ppm -o 2k_wild.450x253.12.ppm -outputWidth 450 -info
./ResizeSample  -i 2k_wild.1920x1080.12.ppm -o 2k_wild.3000x1688.12.ppm -outputWidth 3000 -info
./ResizeSample  -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1000x1000.12.padded.ppm -outputWidth 1000 -outputheight 1000 -background 2048,2048,2048 -info

