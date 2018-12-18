#ifndef CONFIGURE_H
#define CONFIGURE_H

/*Partitioning paramters*/
#define FUSED_POINTS_MAX 8
#define FUSED_LAYERS_MAX 32
#define PARTITIONS_W_MAX 6
#define PARTITIONS_H_MAX 6
#define PARTITIONS_MAX 36
#define THREAD_NUM 1
#define DATA_REUSE 0

#define POLL_MODE 1

/*Debugging information for different components*/
#define DEBUG_INFERENCE 0
#define DEBUG_FTP 0
#define DEBUG_MULTI_FTP 0
#define DEBUG_SERIALIZATION 0
#define DEBUG_DEEP_GATEWAY 0
#define DEBUG_DEEP_EDGE 1
#define DEBUG_REQUEST 0

/*Print timing and communication size information*/
#define DEBUG_TIMING 1
#define DEBUG_COMMU_SIZE 1

/*Configuration parameters for DistrIoT*/
#define GATEWAY_PUBLIC_ADDR "192.168.1.12"
#define GATEWAY_LOCAL_ADDR "192.168.1.12"
//#define EDGE_ADDR_LIST    {"192.168.1.9"}
//#define EDGE_ADDR_LIST    {"192.168.1.9", "192.168.1.10"}
#define EDGE_ADDR_LIST    {"192.168.1.9", "192.168.1.10", "192.168.1.11"}
//#define EDGE_ADDR_LIST    {"192.168.1.9", "192.168.1.10", "192.168.1.11", "192.168.1.12"}
//#define EDGE_ADDR_LIST    {"192.168.1.9", "192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14"}
#define MAX_EDGE_NUM 6
#define FRAME_NUM 1

#define LOAD_AWARE 0
#define CPU_USAGE_DELAY  30000 // > 20ms
#define MAX_CPU_LOAD 0.8
#define MAX_MEM_LOAD 0.8

#endif
