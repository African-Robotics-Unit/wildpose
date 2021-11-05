#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.BGGR.pgm -pattern BGGR
./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.pgm -bits 12 -shift 4 -randomize

# 8-bits sample
./DebayerJpegSample -i 2k_wild.1920x1080.BGGR.pgm -o 2k_wild.1920x1080.BGGR.420.jpg -q 95 -s 420 -pattern BGGR -log debayer_jpeg.log -info
./DebayerJpegSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.422.jpg -s 422 -info

# 12-bits sample
./DebayerJpegSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.422.jpg -s 422 -info
