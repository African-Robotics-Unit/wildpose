#!/bin/bash

. ./Configuration.sh


# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.12.ppm -bits 12 -shift 4 -randomize
./ImageConverterSample -i 2k_wild.12.ppm -o 2k_wild.12.pgm -grayscale
./ImageConverterSample -i 2k_wild.12.pgm -o 2k_wild.PTG.12.raw -format ptg12
./ImageConverterSample -i 2k_wild.12.pgm -o 2k_wild.XIMEA.12.raw -format ximea12

./RawImportSample -i 2k_wild.XIMEA.12.raw -o 2k_wild.1920x1080.Ximea.pgm -width 1920 -height 1080 -format ximea12 -info -log raw.log
./RawImportSample -i 2k_wild.PTG.12.raw -o 2k_wild.1920x1080.PTG.pgm -width 1920 -height 1080 -format ptg12 -info
