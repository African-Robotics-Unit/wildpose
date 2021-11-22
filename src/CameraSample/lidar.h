#ifndef LIDAR_H
#define LIDAR_H

#include <condition_variable>
#include <memory>
#include <fstream>
#include <list>
#include <vector>
#include <mutex>
#include "livox_sdk.h"
#include "lvx_file.h"

#define FRAME_RATE 20
#define EXPECTED_IMAGES 50
#define AUTO_EXP_IMAGES 10

extern DeviceItem devices[kMaxLidarCount];
extern LvxFileHandle lvx_file_handler;
extern std::list<LvxBasePackDetail> point_packet_list;
extern std::vector<std::string> broadcast_code_rev;
extern std::condition_variable lidar_arrive_condition;
extern std::condition_variable extrinsic_condition;
extern std::condition_variable point_pack_condition;
extern std::mutex mtx;
extern int lvx_file_save_time;
extern bool is_finish_extrinsic_parameter;
extern bool is_read_extrinsic_from_xml;
extern uint8_t connected_lidar_count;

/** Connect all the broadcast device in default and connect specific device when use program options or broadcast_code_list is not empty. */
static std::vector<std::string> broadcast_code_list = {
  //"1PQDH5B00103151"
  //"000000000000002",
  //"000000000000003",
  //"000000000000004"
};

void OnLidarErrorStatusCallback(livox_status status, uint8_t handle, ErrorMessage *message);
void GetLidarData(uint8_t handle, LivoxEthPacket *data, uint32_t data_num, void *client_data);
void OnSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data);
void OnStopSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data);
void OnGetLidarExtrinsicParameter(livox_status status, uint8_t handle, LidarGetExtrinsicParameterResponse *response, void *data);
void LidarGetExtrinsicFromXml(uint8_t handle);
void OnDeviceInformation(livox_status status, uint8_t handle, DeviceInformationResponse *ack, void *data);
void LidarConnect(const DeviceInfo *info);
void LidarDisConnect(const DeviceInfo *info);
void LidarStateChange(const DeviceInfo *info);
void OnDeviceInfoChange(const DeviceInfo *info, DeviceEvent type);
void OnDeviceBroadcast(const BroadcastDeviceInfo *info);
void WaitForDevicesReady();
void WaitForExtrinsicParameter();
void AddDevicesToConnect();
void SetProgramOption(int argc, const char *argv[]);

#endif // LIDAR_H
