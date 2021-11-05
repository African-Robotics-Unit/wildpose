#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.16.ppm -bits 16 -shift 8 -randomize
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.pgm -grayscale
./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.12.pgm -bits 12 -shift 4 -randomize

# 8-bit sample (color)
./NppSample -gauss -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.gauss.ppm -sigma 0.95 -radius 1.0 -log nppGauss.log
./NppSample -resize -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.0.5.linear.ppm -resizedWidth 1000 -shift 0.5 -interpolationType linear -log nppResize.log
./NppSample -rotate -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.45.cubic.rotated.ppm -angle 45 -interpolationType cubic -log nppRotate.log
./NppSample -unsharp -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.unsharp.ppm -sigma 0.95 -amount 0.95
./NppSample -unsharp -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.unsharp.threshold.ppm -sigma 0.95 -amount 0.95 -threshold 0.1

# 8-bit sample (gray)
./NppSample -unsharp -i 2k_wild.1920x1080.pgm -o 2k_wild.unsharp.pgm -sigma 0.95 -amount 0.95

# 16-bit sample (color)
./NppSample -remap -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.remap.90.ppm -rotate90 -log nppRemap.log
./NppSample -remap -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.remap.Background.90.ppm -rotate90 -log nppRemapBackground.log -R 4095 -G 4095 -B 4095
./NppSample -remap3 -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.remap3.90.ppm -rotate90

# 16-bit sample
./NppSample -perspective -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.Perspective.90.ppm -perspectiveCoeffs {1.0,0,0,0,1.0,0,0,0,1.0} -log nppPerspective.log
./NppSample -perspective -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.Perspective.pgm

# 12-bit sample (gray)
./NppSample -unsharp -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.unsharp.pgm -sigma 0.95 -amount 0.95 -info
