#!/bin/bash

. ./Configuration.sh
# Prepare data

./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.BGGR.pgm -pattern BGGR
./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.pgm -bits 12 -shift 4 -randomize

# 8-bit
./BayerCompressionSample -i 2k_wild.1920x1080.BGGR.pgm -o 2k_wild.1920x1080.BGGR.jpg -pattern BGGR -s 444 -log bayer.log
./BayerCompressionSample -i 2k_wild.1920x1080.BGGR.jpg -o 2k_wild.1920x1080.BGGR.Restored.pgm

# 12-bit
./BayerCompressionSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.jpg -q 90
./BayerCompressionSample -i 2k_wild.1920x1080.12.RGGB.jpg -o 2k_wild.1920x1080.12.RGGB.Restored.pgm
./BayerCompressionSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.Quant.jpg -quantTable $DATA_SET/QuantTable/QuantTable10.txt
./BayerCompressionSample -i 2k_wild.1920x1080.12.RGGB.Quant.jpg -o 2k_wild.1920x1080.12.RGGB.Quant.Restored.pgm

