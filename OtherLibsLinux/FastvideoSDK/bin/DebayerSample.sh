#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.BGGR.pgm -pattern BGGR
./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.pgm -bits 12 -shift 4 -randomize
./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.16.RGGB.pgm -bits 16 -shift 8 -randomize

./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.MatrixA.1.1.pfm -matrix -pixel 1.1

# 8-bits sample
./DebayerSample -i 2k_wild.1920x1080.BGGR.pgm -o 2k_wild.1920x1080.BGGR.Default.ppm -pattern BGGR -wb_r 0.5 -wb_g1 0.5 -wb_g2 0.5 -wb_b 0.5 -log debayer.log -info
./DebayerSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.HQLI.ppm -type HQLI -info
./DebayerSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.MatrixA.1.1.ppm -matrixA 2k_wild.1920x1080.MatrixA.1.1.pfm -info

# 12-bits sample
./DebayerSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.HQLI.ppm -type HQLI -info
./DebayerSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.DFPD.ppm -type DFPD -info
./DebayerSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.MG.ppm -type MG -info

# 16-bits sample
./DebayerSample -i 2k_wild.1920x1080.16.RGGB.pgm -o 2k_wild.1920x1080.16.RGGB.Binning2x2.ppm -type binning_2x2 -info
./DebayerSample -i 2k_wild.1920x1080.16.RGGB.pgm -o 2k_wild.1920x1080.16.RGGB.Binning4x4.ppm -type binning_4x4 -info
./DebayerSample -i 2k_wild.1920x1080.16.RGGB.pgm -o 2k_wild.1920x1080.16.RGGB.Binning8x8.ppm -type binning_8x8 -info

./DebayerSample -i 2k_wild.1920x1080.16.RGGB.pgm -o 2k_wild.1920x1080.16.RGGB.L7.ppm -type L7 -info

# Multi thread sample
./DebayerSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.DFPD.MultiThread.ppm -pattern RGGB -type DFPD -async -repeat 64 -thread 2 -threadR 2 -threadW 1 -b 2 -info -log MtDebayerSampleTrace.log