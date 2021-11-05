#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.pgm -grayscale
./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.12.pgm -bits 12 -shift 4 -randomize
./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.16.pgm -bits 16 -shift 8 -randomize

./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.16.ppm -bits 16 -shift 8 -randomize

./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB
./ImageConverterSample -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.RGGB.pgm -pattern RGGB
./ImageConverterSample -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.RGGB.pgm -pattern RGGB

./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.MatrixA.1.1.pfm -matrix -pixel 1.1
./ImageConverterSample -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.MatrixB.255.pgm -matrix -pixel 255
./ImageConverterSample -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.MatrixB.4095.pgm -matrix -pixel 4095

# Affine transform (8 bits)
./ComponentsSample -affine flip -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.flip.ppm -info -log flip.log
./ComponentsSample -affine flop -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.flop.ppm -info
./ComponentsSample -affine rotate180 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.rotate180.ppm -info
./ComponentsSample -affine rotate90left -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.rotate90left.ppm -info
./ComponentsSample -affine rotate90right -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.rotate90right.ppm -info

# Affine transform (12 bits)
./ComponentsSample -affine rotate90left -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.rotate90left.pgm -info
./ComponentsSample -affine rotate90left_f -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.rotate90left_f.pgm -info
./ComponentsSample -affine rotate90right -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.rotate90right.pgm -info
./ComponentsSample -affine rotate90right_f -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.rotate90right_f.pgm -info

# LUTs
./ComponentsSample -lut8c -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.Lut8c.ppm -lut_r $DATA_SET/LUT/InvertColor.8bit.txt -lut_g $DATA_SET/LUT/InvertColor.8bit.txt -lut_b $DATA_SET/LUT/InvertColor.8bit.txt -info
./ComponentsSample -lut8_16b -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild.1920x1080.12.RGGB.08.16.pgm -pattern RGGB -lut_r $DATA_SET/LUT/lut12.txt -lut_g $DATA_SET/LUT/lut12.txt -lut_b $DATA_SET/LUT/lut12.txt -info
./ComponentsSample -rgbLut3D -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.LUT3D.ppm -lut $DATA_SET/LUT/lut3D_inverse_12.cube
./ComponentsSample -hsvLut3D -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.HSV3D.ppm -lut $DATA_SET/LUT/HsvLut3D.xml
./ComponentsSample -hsvLut3D -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.HSV2D.ppm -lut $DATA_SET/LUT/HsvLut2D.xml

# SAM
./ComponentsSample -sam -i 2k_wild.1920x1080.pgm -o 2k_wild.MatrixA.1.1.pgm -matrixA 2k_wild.1920x1080.MatrixA.1.1.pfm -info
./ComponentsSample -sam -i 2k_wild.1920x1080.pgm -o 2k_wild.MatrixB.pgm -matrixB 2k_wild.1920x1080.MatrixB.255.pgm -info
./ComponentsSample -sam16 -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.MatrixB.pgm -matrixB 2k_wild.1920x1080.12.MatrixB.4095.pgm -info

# Bad Pixel Correction
./ComponentsSample -badPixelCorrection -i $DATA_SET/Images/BPC.pgm -o BPC.12.RGGB.BPC.pgm -info -pattern RGGB
./ComponentsSample -badPixelCorrection -i $DATA_SET/Images/BPC.pgm -o BPC.12.BPC.pgm -info -pattern none

# Other transfrom (8 bits)
./ComponentsSample -baseColorCorrection -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.BaseColorCorrection.ppm -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -whiteLevel {255,255,255} -info
./ComponentsSample -crop 500x500+100+100 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.500x500.ppm
./ComponentsSample -colorConvertion -i 2k_wild.1920x1080.pgm -o 2k_wild.1920x1080.ColorConvertion.ppm -info
./ComponentsSample -bgrxImport -i $DATA_SET/Images/2k_wild.ppm  -o 2k_wild.BGRX.ppm

./ComponentsSample -sharp -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.sharped.ppm -sigma 0.95
./ComponentsSample -blur -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.blured.ppm -sigma 0.95

# Other transfrom (12/16 bits)
./ComponentsSample -toneCurve $DATA_SET/Other/toneCurve.txt -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.toneCurve.ppm -info
./ComponentsSample -binning -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.binning.avg.2.pgm -mode avg -factor 2
./ComponentsSample -binning -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.binning.sum.2.pgm -mode sum -factor 2
./ComponentsSample -defringe -i $DATA_SET/Images/cakePlus.12.ppm -o cakePlus.12.Defringe.ppm -window 40 -tintR 110 -tintG 113 -tintB 108 coefficient 0.05 -fi_max 60
./ComponentsSample -bitDepthConverter -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12_08.ppm -dstBitsPerChannel 8 -info
./ComponentsSample -bitDepthConverter -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16_08.ppm -dstBitsPerChannel 8 -info
./ComponentsSample -colorConvertion -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.ColorConvertion.pgm -info
./ComponentsSample -median -i 2k_wild.1920x1080.12.ppm -o 2k_wild.1920x1080.12.filtered.ppm -info
./ComponentsSample -median -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.filtered.ppm -info
./ComponentsSample -bayerBlackShift -i 2k_wild.1920x1080.12.pgm -o 2k_wild.1920x1080.12.BlackShift.pgm -blackShiftR 256 -blackShiftG 256 -blackShiftB 256 -info
./ComponentsSample -selectChannel -i 2k_wild.1920x1080.16.ppm -o 2k_wild.1920x1080.16.B.pgm -channel B -info
