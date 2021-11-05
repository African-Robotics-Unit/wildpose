#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.MatrixA.1.1.pfm -matrix -pixel 1.1

./MuxSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.MatrixA.1.1.ppm -matrixA 2k_wild.1920x1080.MatrixA.1.1.pfm -info -log mux.log
