/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#include "RawProcessor.h"
#include "CUDAProcessorBase.h"
#include "CUDAProcessorGray.h"
#include "FrameBuffer.h"
#include "GPUCameraBase.h"
#include "MainWindow.h"
#include "FPNReader.h"
#include "FFCReader.h"
#include "test_encoder.h"
#include <JetsonGPIO.h>
#include <chrono>

#include <algorithm>
#include <string.h>
#include "lvx_file.h"
#include "cmdline.h"

#include "avfilewriter/avfilewriter.h"

#include <QElapsedTimer>
#include <QDateTime>
#include <QDebug>
#include <QPoint>

using namespace std;
using namespace std::chrono;
using namespace GPIO;


test_encoder rot;

DeviceItem devices[kMaxLidarCount];
LvxFileHandle lvx_file_handler;
std::list<LvxBasePackDetail> point_packet_list;
std::vector<std::string> broadcast_code_rev;
std::condition_variable lidar_arrive_condition;
std::condition_variable extrinsic_condition;
std::condition_variable point_pack_condition;
std::mutex mtx;
int lvx_file_save_time = 10;
bool is_finish_extrinsic_parameter = false;
bool is_read_extrinsic_from_xml = false;
uint8_t connected_lidar_count = 0;

#define FRAME_RATE 20

/** Connect all the broadcast device in default and connect specific device when use program options or broadcast_code_list is not empty. */
std::vector<std::string> broadcast_code_list = {
  //"000000000000001"
  //"000000000000002",
  //"000000000000003",
  //"000000000000004"
};

/** Receiving error message from Livox Lidar. */
void OnLidarErrorStatusCallback(livox_status status, uint8_t handle, ErrorMessage *message) {
  static uint32_t error_message_count = 0;
  if (message != NULL) {
    ++error_message_count;
    if (0 == (error_message_count % 100)) {
      printf("handle: %u\n", handle);
      printf("temp_status : %u\n", message->lidar_error_code.temp_status);
      printf("volt_status : %u\n", message->lidar_error_code.volt_status);
      printf("motor_status : %u\n", message->lidar_error_code.motor_status);
      printf("dirty_warn : %u\n", message->lidar_error_code.dirty_warn);
      printf("firmware_err : %u\n", message->lidar_error_code.firmware_err);
      printf("pps_status : %u\n", message->lidar_error_code.device_status);
      printf("fan_status : %u\n", message->lidar_error_code.fan_status);
      printf("self_heating : %u\n", message->lidar_error_code.self_heating);
      printf("ptp_status : %u\n", message->lidar_error_code.ptp_status);
      printf("time_sync_status : %u\n", message->lidar_error_code.time_sync_status);
      printf("system_status : %u\n", message->lidar_error_code.system_status);
    }
  }
}

/** Receiving point cloud data from Livox LiDAR. */
void GetLidarData(uint8_t handle, LivoxEthPacket *data, uint32_t data_num, void *client_data) {
  if (data) {
    if (handle < connected_lidar_count && is_finish_extrinsic_parameter) {
      std::unique_lock<std::mutex> lock(mtx);
      LvxBasePackDetail packet;
      packet.device_index = handle;
      lvx_file_handler.BasePointsHandle(data, packet);
      point_packet_list.push_back(packet);
    }
  }
}

/** Callback function of starting sampling. */
void OnSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data) {
  printf("OnSampleCallback statues %d handle %d response %d \n", status, handle, response);
  if (status == kStatusSuccess) {
    if (response != 0) {
      devices[handle].device_state = kDeviceStateConnect;
    }
  } else if (status == kStatusTimeout) {
    devices[handle].device_state = kDeviceStateConnect;
  }
}

/** Callback function of stopping sampling. */
void OnStopSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data) {
}

/** Callback function of get LiDARs' extrinsic parameter. */
void OnGetLidarExtrinsicParameter(livox_status status, uint8_t handle, LidarGetExtrinsicParameterResponse *response, void *data) {
  if (status == kStatusSuccess) {
    if (response != 0) {
      printf("OnGetLidarExtrinsicParameter statue %d handle %d response %d \n", status, handle, response->ret_code);
      std::unique_lock<std::mutex> lock(mtx);
      LvxDeviceInfo lidar_info;
      strncpy((char *)lidar_info.lidar_broadcast_code, devices[handle].info.broadcast_code, kBroadcastCodeSize);
      memset(lidar_info.hub_broadcast_code, 0, kBroadcastCodeSize);
      lidar_info.device_index = handle;
      lidar_info.device_type = devices[handle].info.type;
      lidar_info.extrinsic_enable = true;
      lidar_info.pitch = response->pitch;
      lidar_info.roll = response->roll;
      lidar_info.yaw = response->yaw;
      lidar_info.x = static_cast<float>(response->x / 1000.0);
      lidar_info.y = static_cast<float>(response->y / 1000.0);
      lidar_info.z = static_cast<float>(response->z / 1000.0);
      lvx_file_handler.AddDeviceInfo(lidar_info);
      if (lvx_file_handler.GetDeviceInfoListSize() == connected_lidar_count) {
        is_finish_extrinsic_parameter = true;
        extrinsic_condition.notify_one();
      }
    }
  }
  else if (status == kStatusTimeout) {
    printf("GetLidarExtrinsicParameter timeout! \n");
  }
}

/** Get LiDARs' extrinsic parameter from file named "extrinsic.xml". */
void LidarGetExtrinsicFromXml(uint8_t handle) {
  LvxDeviceInfo lidar_info;
  ParseExtrinsicXml(devices[handle], lidar_info);
  lvx_file_handler.AddDeviceInfo(lidar_info);
  lidar_info.extrinsic_enable = true;
  if (lvx_file_handler.GetDeviceInfoListSize() == broadcast_code_list.size()) {
    is_finish_extrinsic_parameter = true;
    extrinsic_condition.notify_one();
  }
}

/** Query the firmware version of Livox LiDAR. */
void OnDeviceInformation(livox_status status, uint8_t handle, DeviceInformationResponse *ack, void *data) {
  if (status != kStatusSuccess) {
    printf("Device Query Informations Failed %d\n", status);
  }
  if (ack) {
    printf("firm ver: %d.%d.%d.%d\n",
           ack->firmware_version[0],
           ack->firmware_version[1],
           ack->firmware_version[2],
           ack->firmware_version[3]);
  }
}

void LidarConnect(const DeviceInfo *info) {
  uint8_t handle = info->handle;
  QueryDeviceInformation(handle, OnDeviceInformation, NULL);
  if (devices[handle].device_state == kDeviceStateDisconnect) {
    devices[handle].device_state = kDeviceStateConnect;
    devices[handle].info = *info;
  }
}

void LidarDisConnect(const DeviceInfo *info) {
  uint8_t handle = info->handle;
  devices[handle].device_state = kDeviceStateDisconnect;
}

void LidarStateChange(const DeviceInfo *info) {
  uint8_t handle = info->handle;
  devices[handle].info = *info;
}

/** Callback function of changing of device state. */
void OnDeviceInfoChange(const DeviceInfo *info, DeviceEvent type) {
  if (info == nullptr) {
    return;
  }
  printf("OnDeviceChange broadcast code %s update type %d\n", info->broadcast_code, type);
  uint8_t handle = info->handle;
  if (handle >= kMaxLidarCount) {
    return;
  }

  if (type == kEventConnect) {
    LidarConnect(info);
    printf("[WARNING] Lidar sn: [%s] Connect!!!\n", info->broadcast_code);
  } else if (type == kEventDisconnect) {
    LidarDisConnect(info);
    printf("[WARNING] Lidar sn: [%s] Disconnect!!!\n", info->broadcast_code);
  } else if (type == kEventStateChange) {
    LidarStateChange(info);
    printf("[WARNING] Lidar sn: [%s] StateChange!!!\n", info->broadcast_code);
  }

  if (devices[handle].device_state == kDeviceStateConnect) {
    printf("Device Working State %d\n", devices[handle].info.state);
    if (devices[handle].info.state == kLidarStateInit) {
      printf("Device State Change Progress %u\n", devices[handle].info.status.progress);
    } else {
      printf("Device State Error Code 0X%08x\n", devices[handle].info.status.status_code.error_code);
    }
    printf("Device feature %d\n", devices[handle].info.feature);
    SetErrorMessageCallback(handle, OnLidarErrorStatusCallback);
    if (devices[handle].info.state == kLidarStateNormal) {
      if (!is_read_extrinsic_from_xml) {
        LidarGetExtrinsicParameter(handle, OnGetLidarExtrinsicParameter, nullptr);
      } else {
        LidarGetExtrinsicFromXml(handle);
      }
      LidarStartSampling(handle, OnSampleCallback, nullptr);
      devices[handle].device_state = kDeviceStateSampling;
    }
  }
}

/** Callback function when broadcast message received.
 * You need to add listening device broadcast code and set the point cloud data callback in this function.
 */
void OnDeviceBroadcast(const BroadcastDeviceInfo *info) {
  if (info == nullptr || info->dev_type == kDeviceTypeHub) {
    return;
  }

  printf("Receive Broadcast Code %s\n", info->broadcast_code);
  if ((broadcast_code_rev.size() == 0) ||
      (std::find(broadcast_code_rev.begin(), broadcast_code_rev.end(), info->broadcast_code) == broadcast_code_rev.end())) {
    broadcast_code_rev.push_back(info->broadcast_code);
    lidar_arrive_condition.notify_one();
  }
}

/** Wait until no new device arriving in 2 second. */
void WaitForDevicesReady( ) {
  bool device_ready = false;
  seconds wait_time = seconds(2);
  steady_clock::time_point last_time = steady_clock::now();
  while (!device_ready) {
    std::unique_lock<std::mutex> lock(mtx);
    lidar_arrive_condition.wait_for(lock,wait_time);
    if ((steady_clock::now() - last_time + milliseconds(50)) >= wait_time) {
      device_ready = true;
    } else {
      last_time = steady_clock::now();
    }
  }
}

void WaitForExtrinsicParameter() {
  std::unique_lock<std::mutex> lock(mtx);
  extrinsic_condition.wait(lock);
}

void AddDevicesToConnect() {
  if (broadcast_code_rev.size() == 0)
    return;

  for (int i = 0; i < broadcast_code_rev.size(); ++i) {
    if ((broadcast_code_list.size() != 0) &&
        (std::find(broadcast_code_list.begin(), broadcast_code_list.end(), broadcast_code_rev[i]) == broadcast_code_list.end())) {
      continue;
    }
    uint8_t handle = 0;
    livox_status result = AddLidarToConnect(broadcast_code_rev[i].c_str(), &handle);
    if (result == kStatusSuccess) {
      /** Set the point cloud data for a specific Livox LiDAR. */
      SetDataCallback(handle, GetLidarData, nullptr);
      devices[handle].handle = handle;
      devices[handle].device_state = kDeviceStateDisconnect;
      connected_lidar_count++;
    }
  }
}

/** Set the program options.
* You can input the registered device broadcast code and decide whether to save the log file.
*/
void SetProgramOption(int argc, const char *argv[]) {
  cmdline::parser cmd;
  cmd.add<std::string>("code", 'c', "Register device broadcast code", false);
  cmd.add("log", 'l', "Save the log file");
  cmd.add<int>("time", 't', "Time to save point cloud to the lvx file", false);
  cmd.add("param", 'p', "Get the extrinsic parameter from extrinsic.xml file");
  cmd.add("help", 'h', "Show help");
  cmd.parse_check(argc, const_cast<char **>(argv));
  if (cmd.exist("code")) {
    std::string sn_list = cmd.get<std::string>("code");
    printf("Register broadcast code: %s\n", sn_list.c_str());
    size_t pos = 0;
    broadcast_code_list.clear();
    while ((pos = sn_list.find("&")) != std::string::npos) {
      broadcast_code_list.push_back(sn_list.substr(0, pos));
      sn_list.erase(0, pos + 1);
    }
    broadcast_code_list.push_back(sn_list);
  }
  if (cmd.exist("log")) {
    printf("Save the log file.\n");
    SaveLoggerFile();
  }
  if (cmd.exist("time")) {
    printf("Time to save point cloud to the lvx file:%d.\n", cmd.get<int>("time"));
    lvx_file_save_time = cmd.get<int>("time");
  }
  if (cmd.exist("param")) {
    printf("Get the extrinsic parameter from extrinsic.xml file.\n");
    is_read_extrinsic_from_xml = true;
  }
  return;
}

RawProcessor::RawProcessor(GPUCameraBase *camera, GLRenderer *renderer):QObject(nullptr),
    mCamera(camera),
    mRenderer(renderer)
{
    if(camera->isColor())
        mProcessorPtr.reset(new CUDAProcessorBase());
    else
        mProcessorPtr.reset(new CUDAProcessorGray());

    connect(mProcessorPtr.data(), SIGNAL(error()), this, SIGNAL(error()));

    mCUDAThread.setObjectName(QStringLiteral("CUDAThread"));
    moveToThread(&mCUDAThread);
    mCUDAThread.start();
}

RawProcessor::~RawProcessor()
{
    stop();
    mCUDAThread.quit();
    mCUDAThread.wait(3000);
}

fastStatus_t RawProcessor::init()
{
    if(!mProcessorPtr)
        return FAST_INVALID_VALUE;

    return mProcessorPtr->Init(mOptions);
}

void RawProcessor::start()
{
    if(!mProcessorPtr || mCamera == nullptr)
        return;

    QTimer::singleShot(0, this, [this](){startWorking();});
    
    
}

void RawProcessor::stop()
{
    mWorking = false;
    mWaitCond.wakeAll();

    if(mFileWriterPtr)
    {
        mFileWriterPtr->waitFinish();
        mFileWriterPtr->stop();
    }

    //Wait up to 1 sec until mWorking == false
    QTime tm;
    tm.start();
    while(mWorking && tm.elapsed() <= 1000)
    {
        QThread::msleep(100);
    }
}

void RawProcessor::wake()
{
    mWake = true;
    mWaitCond.wakeAll();
}

void RawProcessor::updateOptions(const CUDAProcessorOptions& opts)
{
    if(!mProcessorPtr)
        return;
    QMutexLocker lock(&(mProcessorPtr->mut));
    mOptions = opts;
}

void RawProcessor::startWorking()
{
    mWorking = true;

    qint64 lastTime = 0;
    QElapsedTimer tm;
    tm.start();

    QByteArray buffer;
    buffer.resize(mOptions.Width * mOptions.Height * 4);

    int bpc = GetBitsPerChannelFromSurface(mCamera->surfaceFormat());
    int maxVal = (1 << bpc) - 1;
    QString pgmHeader = QString("P5\n%1 %2\n%3\n").arg(mOptions.Width).arg(mOptions.Height).arg(maxVal);

    mWake = false;

    rot.start();

    printf("Livox SDK initializing.\n");
    /** Initialize Livox-SDK. */
    if (!Init()) {
        return;
    }
    printf("Livox SDK has been initialized.\n");

    LivoxSdkVersion _sdkversion;
    GetLivoxSdkVersion(&_sdkversion);
    printf("Livox SDK version %d.%d.%d .\n", _sdkversion.major, _sdkversion.minor, _sdkversion.patch);

    memset(devices, 0, sizeof(devices));
    SetBroadcastCallback(OnDeviceBroadcast);

/** Set the callback function called when device state change,
 * which means connection/disconnection and changing of LiDAR state.
 */
    SetDeviceStateUpdateCallback(OnDeviceInfoChange);

/** Start the device discovering routine. */
    if (!Start()) {
        Uninit();
        return;
    }
    printf("Start discovering device.\n");

    WaitForDevicesReady();

    AddDevicesToConnect();

    if (connected_lidar_count == 0) {
        printf("No device will be connected.\n");
        Uninit();
        return;
    }

    WaitForExtrinsicParameter();

    printf("Start initialize lvx file.\n");

    if (!lvx_file_handler.InitLvxFile()) {
        Uninit();
        return;
    }

    lvx_file_handler.InitLvxFileHeader();

    

    while(mWorking)
    {
        if(!mWake)
        {
            mWaitMutex.lock();
            mWaitCond.wait(&mWaitMutex);
            mWaitMutex.unlock();
        }
        mWake = false;
        if(!mWorking)
            break;

        if(!mProcessorPtr || mCamera == nullptr)
            continue;

        GPUImage_t* img = mCamera->getFrameBuffer()->getLastImage();
        mProcessorPtr->Transform(img, mOptions);
        if(mRenderer)
        {
            qint64 curTime = tm.elapsed();
/// arm processor cannot show 60 fps
#ifdef __ARM_ARCH
            const qint64 frameTime = 32;
            if(curTime - lastTime >= frameTime)
#endif
            {
                if(mOptions.ShowPicture){
                    mRenderer->loadImage(mProcessorPtr->GetFrameBuffer(), mOptions.Width, mOptions.Height);
                    mRenderer->update();
                }
                lastTime = curTime;

                emit finished();
            }
        }

        /// added sending by rtsp
        if(mOptions.Codec == CUDAProcessorOptions::vcJPG ||
           mOptions.Codec == CUDAProcessorOptions::vcMJPG)
        {
            if(mRtspServer && mRtspServer->isConnected())
            {
                mRtspServer->addFrame(nullptr);
            }
        }
        if(mOptions.Codec == CUDAProcessorOptions::vcH264 || mOptions.Codec == CUDAProcessorOptions::vcHEVC)
        {
            if(mRtspServer && mRtspServer->isConnected())
            {
                unsigned char* data = (uchar*)buffer.data();
                mProcessorPtr->export8bitData((void*)data, true);

                mRtspServer->addFrame(data);
            }
        }

        

        if(mWriting && mFileWriterPtr)
        {
            if(mOptions.Codec == CUDAProcessorOptions::vcJPG ||
               mOptions.Codec == CUDAProcessorOptions::vcMJPG)
            {
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.jpg").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = mFileWriterPtr->bufferSize();
                    task->data = buf;
                    mProcessorPtr->exportJPEGData(task->data, mOptions.JpegQuality, task->size);
                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();

                    rot.write_encoders(mFrameCnt);

                    steady_clock::time_point last_time = steady_clock::now();

                    if(mFrameCnt%6 == 0){
                        //LIDAR capture
                        std::list<LvxBasePackDetail> point_packet_list_temp;
                        {
                        std::unique_lock<std::mutex> lock(mtx);
                        point_pack_condition.wait_for(lock, milliseconds(kDefaultFrameDurationTime) - (steady_clock::now() - last_time));
                        
                        point_packet_list_temp.swap(point_packet_list);
                        last_time = steady_clock::now();
                        }
                        if(point_packet_list_temp.empty()) {
                        printf("Point cloud packet is empty.\n");
                        break;
                        }
                        printf("LVX data from frame %d received.\n", mFrameCnt); // image buffer

                        lvx_file_handler.SaveFrameToLvxFile(point_packet_list_temp);
                        printf("Finish save %d frame to lvx file.\n", mFrameCnt);
                    }

                    mFrameCnt++;
                }
            }
            else if(mOptions.Codec == CUDAProcessorOptions::vcPGM)
            {
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    unsigned w = 0;
                    unsigned h = 0;
                    unsigned pitch = 0;
                    mProcessorPtr->exportRawData(nullptr, w, h, pitch);

                    int sz = pgmHeader.size() + pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.pgm").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = sz;

                    task->data = buf;
                    memcpy(task->data, pgmHeader.toStdString().c_str(), pgmHeader.size());
                    unsigned char* data = task->data + pgmHeader.size();
                    mProcessorPtr->exportRawData((void*)data, w, h, pitch);

                    //Not 8 bit pgm requires big endian byte order
                    if(img->surfaceFmt != FAST_I8)
                    {
                        unsigned short* data16 = (unsigned short*)data;
                        for(unsigned i = 0; i < w * h; i++)
                        {
                            unsigned short val = *data16;
                            *data16 = (val << 8) | (val >> 8);
                            data16++;
                        }
                    }

                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
                    mFrameCnt++;
                }

            }else if(mOptions.Codec == CUDAProcessorOptions::vcH264 || mOptions.Codec == CUDAProcessorOptions::vcHEVC)
            {
                //unsigned char* buf = mFileWriterPtr->getBuffer();
                /*if(buf != nullptr)*/{
//                    unsigned char* data = (uchar*)buffer.data();
//                    mProcessorPtr->export8bitData((void*)data, true);

//                    int w = mOptions.Width;
//                    int h = mOptions.Height;
//                    int pitch = w * (mProcessorPtr->isGrayscale()? 1 : 3);
//                    int sz = pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.mkv").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = 0;

                    task->data = nullptr;
                    //memcpy(task->data, data, sz);

                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();

                    //mRtspServer->addFrame(data);
                }
            }
        }
    }
    mWorking = false;
}

fastStatus_t RawProcessor::getLastError()
{
    if(mProcessorPtr)
        return mProcessorPtr->getLastError();
    else
        return FAST_OK;
}

QString RawProcessor::getLastErrorDescription()
{
    return  (mProcessorPtr) ? mProcessorPtr->getLastErrorDescription() : QString();
}

QMap<QString, float> RawProcessor::getStats()
{
    QMap<QString, float> ret;
    if(mProcessorPtr)
    {
        {
            // to minimize delay in main thread
            mProcessorPtr->mut2.lock();
            ret = mProcessorPtr->stats2;
            mProcessorPtr->mut2.unlock();
        }

        if(mWriting)
        {
            ret[QStringLiteral("procFrames")] = mFileWriterPtr->getProcessedFrames();
            ret[QStringLiteral("droppedFrames")] = mFileWriterPtr->getDroppedFrames();
            AVFileWriter *obj = dynamic_cast<AVFileWriter*>(mFileWriterPtr.data());
            if(obj)
                ret[QStringLiteral("encoding")] = obj->duration();
        }
        else
        {
            ret[QStringLiteral("procFrames")] = -1;
            ret[QStringLiteral("droppedFrames")] = -1;
        }
        ret[QStringLiteral("acqTime")] = acqTimeNsec;

        if(mRtspServer){
            ret[QStringLiteral("encoding")] = mRtspServer->duration();
        }
    }

    return ret;
}

void RawProcessor::startWriting()
{
    if(mCamera == nullptr)
        return;

    mWriting = false;
    if(QFileInfo(mOutputPath).exists())
    {
        QDir dir;
        if(!dir.mkpath(mOutputPath))
            return;
    }

    if(!QFileInfo(mOutputPath).isDir())
        return;

    mCodec = mOptions.Codec;

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        QString fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/%2.avi").
                    arg(mOutputPath).
                    arg(QDateTime::currentDateTime().toString(QStringLiteral("dd_MM_yyyy_hh_mm_ss"))));
        AsyncMJPEGWriter* writer = new AsyncMJPEGWriter();
        writer->open(mCamera->width(),
                     mCamera->height(),
                     25,
                     mCamera->isColor() ? mOptions.JpegSamplingFmt : FAST_JPEG_Y,
                     fileName);
        mFileWriterPtr.reset(writer);
    }
    else if(mCodec == CUDAProcessorOptions::vcH264 || mCodec == CUDAProcessorOptions::vcHEVC){
        AVFileWriter *writer = new AVFileWriter();

        auto funEncodeNv12 = [this](unsigned char* yuv, int ){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportNV12DataDevice(yuv);
        };
        writer->setEncodeNv12Fun(funEncodeNv12);

#ifdef __ARM_ARCH
        auto funEncodeYuv = [this](unsigned char* yuv, int bitdepth){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            if(bitdepth == 8)
                mProcessorPtr->exportYuv8Data(yuv);
            else
                mProcessorPtr->exportP010Data(yuv);
        };
        writer->setEncodeYUV420Fun(funEncodeYuv);
#else
        auto funEncodeP010 = [this](unsigned char* yuv, int){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportP010DataDevice(yuv);
        };
        writer->setEncodeYUV420Fun(funEncodeP010);
#endif

        writer->open(mCamera->width(),
                     mCamera->height(),
                     mOptions.bitrate,
                     60,
                     mCodec == CUDAProcessorOptions::vcHEVC);
        mFileWriterPtr.reset(writer);
    }
    else
        mFileWriterPtr.reset(new AsyncFileWriter());

    unsigned pitch = 3 *(((mOptions.Width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
    unsigned sz = pitch * mOptions.Height;
    mFileWriterPtr->initBuffers(sz);
    
    rot.open_csv();

    
    
    
    mFrameCnt = 0;
    mWriting = true;
}

void RawProcessor::stopWriting()
{
    mWriting = false;
    if(!mFileWriterPtr)
    {
        mCodec = CUDAProcessorOptions::vcNone;
        return;
    }

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        AsyncMJPEGWriter* writer = static_cast<AsyncMJPEGWriter*>(mFileWriterPtr.data());
        writer->close();
    }
    if(mCodec == CUDAProcessorOptions::vcH264 || mCodec == CUDAProcessorOptions::vcHEVC){
        AVFileWriter *writer = static_cast<AVFileWriter*>(mFileWriterPtr.data());
        writer->close();
    }

    mCodec = CUDAProcessorOptions::vcNone;

    rot.stop();
    rot.close_file();

    lvx_file_handler.CloseLvxFile();

    for (int i = 0; i < kMaxLidarCount; ++i) {
    if (devices[i].device_state == kDeviceStateSampling) 
        {
        /** Stop the sampling of Livox LiDAR. */
            LidarStopSampling(devices[i].handle, OnStopSampleCallback, nullptr);
        }
    }

    GPIO::cleanup();

/** Uninitialize Livox-SDK. */
    Uninit();
    printf("Done\n");
    usleep(10000);
}

void RawProcessor::setSAM(const QString& fpnFileName, const QString& ffcFileName)
{
    FPNReader* fpnReader = gFPNStore->getReader(fpnFileName);

    if(fpnReader)
    {
        auto bpp = GetBitsPerChannelFromSurface(mOptions.SurfaceFmt);
        if(fpnReader->width() != mOptions.Width ||
           fpnReader->height() != mOptions.Height ||
           fpnReader->bpp() != bpp)
        {
            mOptions.MatrixB = nullptr;
        }
        else
        {
            mOptions.MatrixB = fpnReader->data();
        }
    }
    else
        mOptions.MatrixB = nullptr;


    FFCReader* ffcReader = gFFCStore->getReader(ffcFileName);
    if(ffcReader)
    {
        if(ffcReader->width() != mOptions.Width ||
           ffcReader->height() != mOptions.Height)
        {
            mOptions.MatrixA = nullptr;
        }
        else
        {
            mOptions.MatrixA = ffcReader->data();
        }
    }
    else
        mOptions.MatrixA = nullptr;

    init();
}

QColor RawProcessor::getAvgRawColor(QPoint rawPoint)
{
    QColor retClr = QColor(Qt::white);

    if(!mProcessorPtr)
        return retClr;

    qDebug() << rawPoint;

    unsigned int w = 0;
    unsigned int h = 0;
    unsigned int pitch = 0;
    fastStatus_t ret = FAST_OK;

    {
        QMutexLocker locker(&(mProcessorPtr->mut));
        ret =  mProcessorPtr->exportLinearizedRaw(nullptr, w, h, pitch);
    }

    std::unique_ptr<unsigned char, FastAllocator> linearBits16;
    FastAllocator allocator;
    size_t sz = pitch * h * sizeof(unsigned short);

    try
    {
        linearBits16.reset(static_cast<unsigned char*>(allocator.allocate(sz)));
    }
    catch(...)
    {
        return retClr;
    }

    {
        QMutexLocker locker(&(mProcessorPtr->mut));
        ret =  mProcessorPtr->exportLinearizedRaw(linearBits16.get(), w, h, pitch);
    }
    if(ret != FAST_OK)
        return retClr;

    int pickerSize = 4;

    if(rawPoint.x() % 2 != 0)
        rawPoint.rx()--;

    if(rawPoint.x() < pickerSize)
        rawPoint.setX(pickerSize);
    if(rawPoint.x() >= int(w) - pickerSize)
        rawPoint.setX(int(w) - pickerSize);

    if(rawPoint.y() % 2 != 0)
        rawPoint.ry()--;
    if(rawPoint.y() < pickerSize)
        rawPoint.setY(pickerSize);
    if(rawPoint.y() >= int(h) - pickerSize)
        rawPoint.setY(int(h) - pickerSize);

    int rOffset = 0;
    int g1Offset = 0;
    int g2Offset = 0;
    int bOffset = 0;

    int rowWidth = pitch / sizeof(unsigned short);

    if(mOptions.BayerFormat == FAST_BAYER_RGGB)
    {
        rOffset = 0;
        g1Offset = 1;
        g2Offset = rowWidth;
        bOffset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_BGGR)
    {
        bOffset = 0;
        g1Offset = 1;
        g2Offset = rowWidth;
        rOffset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_GBRG)
    {
        g1Offset = 0;
        bOffset = 1;
        rOffset = rowWidth;
        g2Offset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_GRBG)
    {
        g1Offset = 0;
        rOffset = 1;
        bOffset = rowWidth;
        g2Offset = rowWidth + 1;
    }
    else
        return {};

    int x = rawPoint.x();
    int y = rawPoint.y();
    auto * rawBits = reinterpret_cast<unsigned short*>(linearBits16.get());

    int r = 0;
    int g = 0;
    int b = 0;
    int cnt = 0;

    for(x = rawPoint.x() - pickerSize; x < rawPoint.x() + pickerSize; x += 2)
    {
        for(y = rawPoint.y() - pickerSize; y < rawPoint.y() + pickerSize; y += 2)
        {
            unsigned short* pixelPtr = rawBits + y * rowWidth + x;

            unsigned int val = pixelPtr[rOffset];
            r += val;

            val = pixelPtr[g1Offset] + pixelPtr[g2Offset];
            g += val;

            val = pixelPtr[bOffset];
            b += val;

            cnt++;
        }
    }

    if(cnt > 1)
    {
        r /= cnt;
        g /= 2 * cnt;
        b /= cnt;
    }

    return {qRgba64(quint16(r), quint16(g), quint16(b), 0)};

}

void RawProcessor::setRtspServer(const QString &url)
{
    if(url.isEmpty())
        return;

    if(mOptions.Width == 0 || mOptions.Height == 0){
        return;
    }
    mUrl = url;

    RTSPStreamerServer::EncoderType encType = RTSPStreamerServer::etJPEG;
    if(mOptions.Codec == CUDAProcessorOptions::vcH264)
        encType = RTSPStreamerServer::etNVENC;
    if(mOptions.Codec == CUDAProcessorOptions::vcHEVC)
        encType = RTSPStreamerServer::etNVENC_HEVC;

    mRtspServer.reset(new RTSPStreamerServer(mOptions.Width, mOptions.Height, 3, url, encType, mOptions.bitrate));

    mRtspServer->setMultithreading(false);

	auto funEncode = [this](int, unsigned char* , int width, int height, int, Buffer& output){

		int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

		unsigned pitch = channels *(((width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
        unsigned sz = pitch * height;

		output.buffer.resize(sz);
		mProcessorPtr->exportJPEGData(output.buffer.data(), mOptions.JpegQuality, sz);
		output.size = sz;
    };
    mRtspServer->setUseCustomEncodeJpeg(true);
    mRtspServer->setEncodeFun(funEncode);

    auto funEncodeNv12 = [this](unsigned char* yuv, int ){
        mProcessorPtr->exportNV12DataDevice(yuv);
    };
    mRtspServer->setEncodeNv12Fun(funEncodeNv12);

#ifdef __ARM_ARCH
        auto funEncodeYuv = [this](unsigned char* yuv, int bitdepth){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            if(bitdepth == 8)
                mProcessorPtr->exportYuv8Data(yuv);
            else
                mProcessorPtr->exportP010Data(yuv);
        };
        mRtspServer->setEncodeYUV420Fun(funEncodeYuv);
#else
        auto funEncodeP010 = [this](unsigned char* yuv, int){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportP010DataDevice(yuv);
        };
        mRtspServer->setEncodeYUV420Fun(funEncodeP010);
#endif

    mRtspServer->startServer();
}

QString RawProcessor::url() const
{
    return mUrl;
}

void RawProcessor::startRtspServer()
{
    if(mUrl.isEmpty())
        return;

    if(mCamera == nullptr)
        return;

    mCodec = mOptions.Codec;

    setRtspServer(mUrl);
}

void RawProcessor::stopRtspServer()
{
    mRtspServer.reset();
}

bool RawProcessor::isStartedRtsp() const
{
    return mRtspServer && mRtspServer->isStarted();
}

bool RawProcessor::isConnectedRtspClient() const
{
    return mRtspServer && mRtspServer->isConnected();
}
