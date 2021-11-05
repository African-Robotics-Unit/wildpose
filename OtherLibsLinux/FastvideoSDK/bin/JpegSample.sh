#!/bin/bash

. ./Configuration.sh


# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.pgm -grayscale

# 8-bits sample
./JpegSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.444.q95.jpg -q 95 -s 444 -info
./JpegSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.Default.jpg -info
./JpegSample -i 2k_wild.1920x1080.pgm -o 2k_wild.gray.jpg -info

./JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.ppm -info

# 8-bits sample (Multi-thread)
./JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.r1.mt.ppm -async -thread 2 -repeat 64 -threadR 2 -threadW 2 -info

# 8-bits sample (Other)
./JpegSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.Default.CustomQuantTable.jpg -info -quantTable $DATA_SET/QuantTable/QuantTable10.txt

./JpegSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.BayerCompressed.jpg -bc -info
./JpegSample -i 2k_wild.1920x1080.RGGB.BayerCompressed.jpg -o 2k_wild.1920x1080.RGGB.BayerCompressed.pgm -bc -info

# 12-bits sample
./JpegSample -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.jpg -s 444 -info
