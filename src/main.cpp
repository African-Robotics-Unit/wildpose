//
// The MIT License (MIT)
//
// Copyright (c) 2019 Livox. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <algorithm>
#include <string.h>
#include "lvx_file.h"
#include <cmdline.h>
#include <stdio.h>
#include <JetsonGPIO.h>
#include <memory.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <chrono>
#ifdef WIN32
#include <xiApi.h>       // Windows
#else
#include <m3api/xiApi.h> // Linux, OSX
#include <unistd.h>      // usleep
#endif
#include "../include/xiAPI/xiAPI_tiff.h"

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
#define EXPECTED_IMAGES 50
#define AUTO_EXP_IMAGES 10

// Check error macro. It executes function. Print and throw error if result is not OK.
#define CE(func) {XI_RETURN stat = (func); if (XI_OK!=stat) {printf("Error:%d returned from function:"#func"\n",stat);throw "Error";}}

using namespace std;
using namespace std::chrono;
using namespace GPIO;

/** Connect all the broadcast device in default and connect specific device when use program options or broadcast_code_list is not empty. */
std::vector<std::string> broadcast_code_list = {
  //"000000000000001"
  //"000000000000002",
  //"000000000000003",
  //"000000000000004"
};

std::mutex m;
int counter = 0;

// Tester function for updating global variable in separate thread
int test(const int& value = 0)
{
    std::lock_guard<std::mutex> lock(m);
    if (value == 0)
    {
        return counter;
    }
    else
    {
        counter = value;
        return 0;
    }
}

void encoder1_thread(){
    int encode_a = 15;
    int encode_b = 13;
    int val_a;
    int val_b;
    GPIO::cleanup();
	  GPIO::setmode(GPIO::BOARD);
	  GPIO::setup(encode_a, GPIO::INTO);
    GPIO::setup(encode_b, GPIO::INTO);

    int last_reading;
    int counter = 0;
    string currentDir ="";

    last_reading = GPIO::input(encode_b);

    while (counter<1000){
        // get a sample
        val_a = GPIO::input(encode_a);
        val_b = GPIO::input(encode_b);

        if ((val_b != last_reading) && val_b==1){
            if (val_a != val_b) {
			    counter --;
			    currentDir ="CCW";
		    } else {
			    // Encoder is rotating CW so increment
			    counter ++;
			    currentDir ="CW";
		    }

		    cout << ("Direction: ");
		    cout << (currentDir);
		    cout << (" | Counter: ");
		    cout << (counter) << endl;
	    }

        last_reading = val_b;
        
    }

}

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

int main(int argc, const char *argv[]) {
/** Set the program options. */
  HANDLE xiH = NULL;
  SetProgramOption(argc, argv);

  printf("Livox SDK initializing.\n");
/** Initialize Livox-SDK. */
  if (!Init()) {
    return -1;
  }
  printf("Livox SDK has been initialized.\n");

  LivoxSdkVersion _sdkversion;
  GetLivoxSdkVersion(&_sdkversion);
  printf("Livox SDK version %d.%d.%d .\n", _sdkversion.major, _sdkversion.minor, _sdkversion.patch);

  printf("Opening first camera...\n");
	CE(xiOpenDevice(0, &xiH));

	printf("Setting exposure time to 10ms...\n");
	CE(xiSetParamInt(xiH, XI_PRM_EXPOSURE, 10000));
	CE(xiSetParamInt(xiH,XI_PRM_BUFFER_POLICY,XI_BP_SAFE));
	int is_color = 0;
	CE(xiGetParamInt(xiH, XI_PRM_IMAGE_IS_COLOR, &is_color));
	if (is_color)
	{
		// color camera
		printf("Colours...\n");
		CE(xiSetParamInt(xiH, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB32));
		CE(xiSetParamInt(xiH, XI_PRM_AUTO_WB, XI_ON));
		CE(xiSetParamFloat(xiH, XI_PRM_EXP_PRIORITY, 1));
	}

  memset(devices, 0, sizeof(devices));

/** Set the callback function receiving broadcast message from Livox LiDAR. */
  SetBroadcastCallback(OnDeviceBroadcast);

/** Set the callback function called when device state change,
 * which means connection/disconnection and changing of LiDAR state.
 */
  SetDeviceStateUpdateCallback(OnDeviceInfoChange);

/** Start the device discovering routine. */
  if (!Start()) {
    Uninit();
    return -1;
  }
  printf("Start discovering device.\n");

  WaitForDevicesReady();

  AddDevicesToConnect();

  if (connected_lidar_count == 0) {
    printf("No device will be connected.\n");
    Uninit();
    return -1;
  }

  WaitForExtrinsicParameter();

  printf("Start initialize lvx file.\n");
  if (!lvx_file_handler.InitLvxFile()) {
    Uninit();
    return -1;
  }

  lvx_file_handler.InitLvxFileHeader();

  int i = 0;
  steady_clock::time_point last_time = steady_clock::now();
  int img_size_bytes = 0;
  CE(xiGetParamInt(xiH, XI_PRM_IMAGE_PAYLOAD_SIZE, &img_size_bytes));
  std::vector<XI_IMG> images;
  for (int i = 0; i< EXPECTED_IMAGES; i++)
  {
    XI_IMG image;
    memset(&image, 0, sizeof(image));
    image.size = sizeof(XI_IMG);
    image.bp = (unsigned char*)malloc(img_size_bytes);
    image.bp_size = img_size_bytes;
    images.push_back(image);
  }
  printf("Starting acquisition...\n");
  xiSetParamInt(xiH, XI_PRM_TRG_SOURCE, XI_TRG_SOFTWARE);
  // set digital output 1 mode
  xiSetParamInt(xiH, XI_PRM_GPO_SELECTOR, 1);
  xiSetParamInt(xiH, XI_PRM_GPO_MODE,  XI_GPO_FRAME_ACTIVE);
  //xiSetParamInt(xiH, XI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT, 500);
	CE(xiStartAcquisition(xiH));
  XI_IMG xi_arr[EXPECTED_IMAGES];


  
  for (i = 0; i < EXPECTED_IMAGES; ++i) {
    std::list<LvxBasePackDetail> point_packet_list_temp;
    {
      std::unique_lock<std::mutex> lock(mtx);
      point_pack_condition.wait_for(lock, milliseconds(kDefaultFrameDurationTime) - (steady_clock::now() - last_time));
      
      point_packet_list_temp.swap(point_packet_list);
      xiSetParamInt(xiH, XI_PRM_TRG_SOFTWARE, 1);
      XI_IMG captured_image = images.at(i);
			CE(xiGetImage(xiH, 150, &captured_image));
			images.at(i) = captured_image;
			unsigned char pixel = *(unsigned char*)captured_image.bp;
			int nframe = captured_image.nframe;
			printf("Image %d received from camera. First pixel value: %d\n", nframe, pixel);
      last_time = steady_clock::now();
    }
    if(point_packet_list_temp.empty()) {
      printf("Point cloud packet is empty.\n");
      break;
    }
    printf("LVX data from frame %d received.\n", i); // image buffer
    //XI_IMG image;
		//memset(&image, 0, sizeof(&image));
		//image.size = sizeof(XI_IMG);
		//CE(xiGetImage(xiH, 100, &image)); // getting next image from the camera opened
		//unsigned char pixel = *(unsigned char*)image.bp;
		//printf("Image %d (%dx%d) received from camera. \n", i, (int)image.width, (int)image.height);
    //xi_arr[i] = image;

    lvx_file_handler.SaveFrameToLvxFile(point_packet_list_temp);
    printf("Finish save %d frame to lvx file.\n", i);
    
		
  }

  lvx_file_handler.CloseLvxFile();
  printf("Stopping acquisition...\n");
  xiStopAcquisition(xiH);
  

  for (int i = 0; i < EXPECTED_IMAGES; ++i)
		{
      char filename[100] = "";
		  sprintf(filename, "image%03d.tif", i);
      XI_IMG img = images.at(i);
		  WriteImage(&img, filename);
		}

  

  for (i = 0; i < kMaxLidarCount; ++i) {
    if (devices[i].device_state == kDeviceStateSampling) {
/** Stop the sampling of Livox LiDAR. */
      
      LidarStopSampling(devices[i].handle, OnStopSampleCallback, nullptr);
      
    }
  }

	

/** Uninitialize Livox-SDK. */
  Uninit();
	printf("Done\n");
  xiCloseDevice(xiH);
  usleep(10000);
}