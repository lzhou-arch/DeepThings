#include "darkiot.h"
#include "configure.h"
#include "cmd_line_parser.h"
#include "deepthings_edge.h"
#include "deepthings_gateway.h"

/*
./deepthings -mode start
./deepthings -mode gateway -total_edge 6 -n 5 -m 5 -l 16
./deepthings -mode data_src -edge_id 0 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 1 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 2 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 3 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 4 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 5 -n 5 -m 5 -l 16

./deepthings -mode <execution mode: {start, gateway, data_src, non_data_src}> 
             -total_edge <total edge number: t> 
             -edge_id <edge device ID: e={0, ... t-1}>
             -n <FTP dimension: N> 
             -m <FTP dimension: M> 
             -l <numder of fused layers: L>
*/

/*"models/yolo.cfg", "models/yolo.weights"*/
static const char* addr_list[MAX_EDGE_NUM] = EDGE_ADDR_LIST;

int main(int argc, char **argv){
   uint32_t total_cli_num = 1;
   uint32_t this_cli_id = 0;

   // set the num of split points
   uint32_t num_sp = get_int_arg(argc, argv, "-sp", 1);
   uint32_t* partitions_h = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* partitions_w = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* from_layers = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* fused_layers = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
  
   partitions_h[0] = get_int_arg(argc, argv, "-n1", 4);
   partitions_w[0] = get_int_arg(argc, argv, "-m1", 4);
   from_layers[0] = get_int_arg(argc, argv, "-f1", 0);
   fused_layers[0] = get_int_arg(argc, argv, "-l1", 16);

   partitions_h[1] = get_int_arg(argc, argv, "-n2", 4);
   partitions_w[1] = get_int_arg(argc, argv, "-m2", 4);
   from_layers[1] = get_int_arg(argc, argv, "-f2", 8);
   fused_layers[1] = get_int_arg(argc, argv, "-l2", 16);

   partitions_h[2] = get_int_arg(argc, argv, "-n3", 4);
   partitions_w[2] = get_int_arg(argc, argv, "-m3", 4);
   from_layers[2] = get_int_arg(argc, argv, "-f3", 8);
   fused_layers[2] = get_int_arg(argc, argv, "-l3", 16);

   partitions_h[3] = get_int_arg(argc, argv, "-n4", 4);
   partitions_w[3] = get_int_arg(argc, argv, "-m4", 4);
   from_layers[3] = get_int_arg(argc, argv, "-f4", 8);
   fused_layers[3] = get_int_arg(argc, argv, "-l4", 16);

   partitions_h[4] = get_int_arg(argc, argv, "-n5", 4);
   partitions_w[4] = get_int_arg(argc, argv, "-m5", 4);
   from_layers[4] = get_int_arg(argc, argv, "-f5", 8);
   fused_layers[4] = get_int_arg(argc, argv, "-l5", 16);

   partitions_h[5] = get_int_arg(argc, argv, "-n6", 4);
   partitions_w[5] = get_int_arg(argc, argv, "-m6", 4);
   from_layers[5] = get_int_arg(argc, argv, "-f6", 8);
   fused_layers[5] = get_int_arg(argc, argv, "-l6", 16);

   for(uint32_t i=0; i<num_sp; i++) {
    fprintf(stderr, "Split points %lu: [%lu, %lu)\n", i, from_layers[i], from_layers[i] + fused_layers[i]); 
   }

   // pick a cnn model
   // small
   //char network_file[30] = "models/alexnet.cfg";
   //char weight_file[30] = "models/alexnet.weights";
   // medium
   //char network_file[30] = "models/yolo.cfg";
   //char weight_file[30] = "models/yolo.weights";
   //
   //char network_file[30] = "models/resnet50.cfg";
   //char weight_file[30] = "models/resnet50.weights";
   //char network_file[30] = "models/resnet152.cfg";
   //char weight_file[30] = "models/resnet152.weights";
   // large
   char network_file[30] = "models/vgg-16.cfg";
   char weight_file[30] = "models/vgg-16.weights";

   if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL + 10, TCP, GATEWAY_PUBLIC_ADDR);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "gateway")){
      printf("Gateway device\n");
      printf("We have %d edge devices now\n", get_int_arg(argc, argv, "-total_edge", 0));
      total_cli_num = get_int_arg(argc, argv, "-total_edge", 0);
      deepthings_gateway(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "data_src")){
      printf("Data source edge device\n");
      printf("This client ID is %d\n", get_int_arg(argc, argv, "-edge_id", 0));
      this_cli_id = get_int_arg(argc, argv, "-edge_id", 0);
      total_cli_num = get_int_arg(argc, argv, "-total_edge", 0);
      deepthings_victim_edge(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "non_data_src")){
      printf("Idle edge device\n");
      printf("This client ID is %d\n", get_int_arg(argc, argv, "-edge_id", 0));
      this_cli_id = get_int_arg(argc, argv, "-edge_id", 0);
      total_cli_num = get_int_arg(argc, argv, "-total_edge", 0);
      deepthings_stealer_edge(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "estimate")) {
      printf("Find the optimal fused points\n");
      deepthings_estimate(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else {
      printf("Invalid cmd.\n");
      exit(-1);
   }
   return 0;
}

