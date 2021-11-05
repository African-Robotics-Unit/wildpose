#!/bin/bash

. ./Configuration.sh

#  Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize

#  8-bits sample
./DenoiseSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CDF53.ppm -w CDF53 -l 4 -t 10 -info -log denoiser.log
./DenoiseSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CDF97.ppm -w CDF97 -l 4 -t 10 -info

#  12-bits sample
./DenoiseSample -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.CDF53.ppm -w CDF53 -l 4 -t 10 -info
./DenoiseSample -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.CDF97.ppm -w CDF97 -l 4 -t 10 -info
