#!/bin/bash

. ./Configuration.sh
#bitdepth 8 bits

./SDIConverterSample -export -format CbYCr422_709 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CbYCr.709.sdi -info
./SDIConverterSample -format CbYCr422_709 -width 1920 -height 1080 -i 2k_wild.CbYCr.709.sdi -o 2k_wild.CbYCr.709.ppm -info

./SDIConverterSample -export -format CbYCr422_2020 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CbYCr.2020.sdi -info
./SDIConverterSample -format CbYCr422_2020 -width 1920 -height 1080 -i 2k_wild.CbYCr.2020.sdi -o 2k_wild.CbYCr.2020.ppm -info

./SDIConverterSample -export -format CbYCr422_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CbYCr.601.sdi -info
./SDIConverterSample -format CbYCr422_601 -width 1920 -height 1080 -i 2k_wild.CbYCr.601.sdi -o 2k_wild.CbYCr.601.ppm -info

./SDIConverterSample -export -format CrYCb422_709 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CrYCb.709.sdi -info
./SDIConverterSample -format CrYCb422_709 -width 1920 -height 1080 -i 2k_wild.CrYCb.709.sdi -o 2k_wild.CrYCb.709.ppm -info

./SDIConverterSample -export -format CrYCb422_2020 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CrYCb.2020.sdi -info
./SDIConverterSample -format CrYCb422_2020 -width 1920 -height 1080 -i 2k_wild.CrYCb.2020.sdi -o 2k_wild.CrYCb.2020.ppm -info

./SDIConverterSample -export -format CrYCb422_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.CrYCb.601.sdi -info
./SDIConverterSample -format CrYCb422_601 -width 1920 -height 1080 -i 2k_wild.CrYCb.601.sdi -o 2k_wild.CrYCb.601.ppm -info

./SDIConverterSample -export -format YV12_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.YV12.601.sdi -info
./SDIConverterSample -format YV12_601 -width 1920 -height 1080 -i 2k_wild.YV12.601.sdi -o 2k_wild.YV12.601.ppm -info

./SDIConverterSample -export -format YV12_709 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.GPU.YV12.709.sdi -gpu -info
./SDIConverterSample -format YV12_709 -width 1920 -height 1080 -i 2k_wild.GPU.YV12.709.sdi -o 2k_wild.GPU.YV12.709.ppm -gpu -info

./SDIConverterSample -export -format YV12_2020 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.GPU.YV12.2020.sdi -gpu -info
./SDIConverterSample -format YV12_2020 -width 1920 -height 1080 -i 2k_wild.GPU.YV12.2020.sdi -o 2k_wild.GPU.YV12.2020.ppm -gpu -info

./SDIConverterSample -export -format YCbCr420_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.YCbCr420.601.sdi -info
./SDIConverterSample -format YCbCr420_601 -width 1920 -height 1080 -i 2k_wild.YCbCr420.601.sdi -o 2k_wild.YCbCr420.601.ppm -info

./SDIConverterSample -export -format RGBA -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.GPU.RGBA.sdi -gpu -info
./SDIConverterSample -format RGBA -width 1920 -height 1080 -i 2k_wild.GPU.RGBA.sdi -o 2k_wild.GPU.RGBA.ppm -gpu -info

./SDIConverterSample -export -format NV12_601_FR -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.GPU.NV12.sdi -gpu -info
./SDIConverterSample -format NV12_601 -width 1920 -height 1080 -i 2k_wild.GPU.NV12.sdi -o 2k_wild.GPU.NV12.ppm -gpu -info

./SDIConverterSample -export -format YCbCr422_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.YCbCr422.601.sdi -info
./SDIConverterSample -format YCbCr422_601 -width 1920 -height 1080 -i 2k_wild.YCbCr422.601.sdi -o 2k_wild.YCbCr422.601.ppm -info

./SDIConverterSample -export -format YCbCr444_601 -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.YCbCr444.601.sdi -info
./SDIConverterSample -format YCbCr444_601 -width 1920 -height 1080 -i 2k_wild.YCbCr444.601.sdi -o 2k_wild.YCbCr444.601.ppm -info


# bitdepth 10/12 bits
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.12.ppm -bits 12 -shift 4 -randomize

./SDIConverterSample -export -format P010_601 -i 2k_wild.1920x1080.12.ppm -o 2k_wild.GPU.P010.601.sdi -gpu -info
./SDIConverterSample -format P010_601 -width 1920 -height 1080 -i 2k_wild.GPU.P010.601.sdi -o 2k_wild.GPU.P010.601.ppm -gpu -info

./SDIConverterSample -export -format CbYCr422_10_601 -i 2k_wild.1920x1080.12.ppm -o 2k_wild.CbYCr422.10.601.sdi -info
./SDIConverterSample -format CbYCr422_10_601 -width 1920 -height 1080 -i 2k_wild.CbYCr422.10.601.sdi -o 2k_wild.CbYCr422.10.601.ppm -info

./SDIConverterSample -export -format YCbCr444_10_601 -i 2k_wild.1920x1080.12.ppm -o 2k_wild.YCbCr444.10.601.sdi -info
./SDIConverterSample -format YCbCr444_10_601 -width 1920 -height 1080 -i 2k_wild.YCbCr444.10.601.sdi -o 2k_wild.YCbCr444.10.601.ppm -info

./SDIConverterSample -export -format YCbCr420_10_601 -i 2k_wild.1920x1080.12.ppm -o 2k_wild.YCbCr420.10.601.sdi -info
./SDIConverterSample -format YCbCr420_10_601 -width 1920 -height 1080 -i 2k_wild.YCbCr420.10.601.sdi -o 2k_wild.YCbCr420.10.601.ppm -info

./SDIConverterSample -export -format YCbCr422_10_601 -i 2k_wild.1920x1080.12.ppm -o 2k_wild.YCbCr422.10.601.sdi -info
./SDIConverterSample -format YCbCr422_10_601 -width 1920 -height 1080 -i 2k_wild.YCbCr422.10.601.sdi -o 2k_wild.YCbCr422.10.601.ppm -info

./SDIConverterSample -export -format BMR10L -i 2k_wild.1920x1080.12.ppm -o 2k_wild.BMR10L.sdi -info
./SDIConverterSample -format BMR10L -width 1920 -height 1080 -i 2k_wild.BMR10L.sdi -o 2k_wild.BMR10L.ppm -info

./SDIConverterSample -export -format BMR10B -i 2k_wild.1920x1080.12.ppm -o 2k_wild.BMR10B.sdi -info
./SDIConverterSample -format BMR10R -width 1920 -height 1080 -i 2k_wild.BMR10B.sdi -o 2k_wild.BMR10B.ppm -info

./SDIConverterSample -export -format BMR12L -i 2k_wild.1920x1080.12.ppm -o 2k_wild.BMR12L.sdi -info
./SDIConverterSample -format BMR12L -width 1920 -height 1080 -i 2k_wild.BMR12L.sdi -o 2k_wild.BMR12L.ppm -info

./SDIConverterSample -export -format BMR12B -i 2k_wild.1920x1080.12.ppm -o 2k_wild.BMR12B.sdi -info
./SDIConverterSample -format BMR12B -width 1920 -height 1080 -i 2k_wild.BMR12B.sdi -o 2k_wild.BMR12B.ppm -info

