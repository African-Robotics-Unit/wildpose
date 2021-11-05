#!/bin/bash

. ./Configuration.sh

./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.ppm
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.Batch.ppm -b 8 -repeat 8
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.BatchMt.ppm -b 8 -repeat 8 -thread 2 -threadR 2 -threadW 2 -async 

./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.window.ppm -window 1000x500+100+100

./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.7bits.ppm -bits 7
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.21passes.ppm -passes 21