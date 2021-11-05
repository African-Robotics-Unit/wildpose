#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.pgm -grayscale
./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.12.pgm -bits 12 -shift 4 -randomize
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize
./ImageConverterSample -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.RGGB.pgm -pattern RGGB

# 8-bits sample
./HistogramSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.hist.color.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype common
./HistogramSample -i 2k_wild.1920x1080.pgm -o 2k_wild.hist.gray.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype common -log hist.log
./HistogramSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.hist.bayer.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype bayer -pattern RGGB
./HistogramSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.RGGB.hist.bayer_g1g2.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype bayer_g1g2 -pattern RGGB
./HistogramSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.parade.color.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype parade -cstride 1

# 12-bits sample
./HistogramSample -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.RGGB.hist.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype common
./HistogramSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.hist.bayer.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype bayer -pattern RGGB
./HistogramSample -i 2k_wild.1920x1080.12.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.hist.bayer_g1g2.txt -bins 256 -sX 0 -sY 0 -roiWidth 1920 -roiHeight 1080 -htype bayer_g1g2 -pattern RGGB

