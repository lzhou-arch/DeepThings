#ifndef CONFIGURE_H
#define CONFIGURE_H

/*Partitioning paramters*/
#define FUSED_LAYERS_MAX 16
#define PARTITIONS_W_MAX 6
#define PARTITIONS_H_MAX 6
#define PARTITIONS_MAX 36
#define THREAD_NUM 1
#define DATA_REUSE 0

/*Debugging information for different components*/
#define DEBUG_INFERENCE 0
#define DEBUG_FTP 0
#define DEBUG_SERIALIZATION 0
#define DEBUG_DEEP_GATEWAY 0
#define DEBUG_DEEP_EDGE 0
#define DEBUG_REQUEST 1

/*Print timing and communication size information*/
#define DEBUG_TIMING 1
#define DEBUG_COMMU_SIZE 0

/*Configuration parameters for DistrIoT*/
#define GATEWAY_PUBLIC_ADDR "192.168.1.9"
#define GATEWAY_LOCAL_ADDR "192.168.1.9"
#define EDGE_ADDR_LIST    {"192.168.1.9", "192.168.1.10"}
//#define EDGE_ADDR_LIST    {"192.168.1.9"}
#define MAX_EDGE_NUM 6
#define FRAME_NUM 1

#endif
