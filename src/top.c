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
             -d <number of devices: D>
*/

/*"models/yolo.cfg", "models/yolo.weights"*/
static const char* addr_list[MAX_EDGE_NUM] = EDGE_ADDR_LIST;

int main(int argc, char **argv){
   uint32_t this_cli_id = 0;

   // set the split points and configs
   uint32_t num_sp = get_int_arg(argc, argv, "-sp", 1);
   uint32_t* partitions_h = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* partitions_w = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* from_layers = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* fused_layers = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
   uint32_t* num_devices = (uint32_t*)malloc(sizeof(uint32_t) * FUSED_POINTS_MAX);
  
   // sp 1
   partitions_h[0] = get_int_arg(argc, argv, "-n1", 2);
   partitions_w[0] = get_int_arg(argc, argv, "-m1", 2);
   from_layers[0] = get_int_arg(argc, argv, "-f1", 0);
   fused_layers[0] = get_int_arg(argc, argv, "-l1", 8);
   num_devices[0] = get_int_arg(argc, argv, "-d1", 1);
   // sp 2
   partitions_h[1] = get_int_arg(argc, argv, "-n2", 2);
   partitions_w[1] = get_int_arg(argc, argv, "-m2", 2);
   from_layers[1] = get_int_arg(argc, argv, "-f2", 8);
   fused_layers[1] = get_int_arg(argc, argv, "-l2", 16);
   num_devices[1] = get_int_arg(argc, argv, "-d2", 1);
   // sp 3
   partitions_h[2] = get_int_arg(argc, argv, "-n3", 2);
   partitions_w[2] = get_int_arg(argc, argv, "-m3", 2);
   from_layers[2] = get_int_arg(argc, argv, "-f3", 8);
   fused_layers[2] = get_int_arg(argc, argv, "-l3", 16);
   num_devices[2] = get_int_arg(argc, argv, "-d3", 1);
   // sp 4
   partitions_h[3] = get_int_arg(argc, argv, "-n4", 2);
   partitions_w[3] = get_int_arg(argc, argv, "-m4", 2);
   from_layers[3] = get_int_arg(argc, argv, "-f4", 8);
   fused_layers[3] = get_int_arg(argc, argv, "-l4", 16);
   num_devices[3] = get_int_arg(argc, argv, "-d4", 1);
   // sp 5
   partitions_h[4] = get_int_arg(argc, argv, "-n5", 2);
   partitions_w[4] = get_int_arg(argc, argv, "-m5", 2);
   from_layers[4] = get_int_arg(argc, argv, "-f5", 8);
   fused_layers[4] = get_int_arg(argc, argv, "-l5", 16);
   num_devices[4] = get_int_arg(argc, argv, "-d5", 1);
   // sp 6
   partitions_h[5] = get_int_arg(argc, argv, "-n6", 2);
   partitions_w[5] = get_int_arg(argc, argv, "-m6", 2);
   from_layers[5] = get_int_arg(argc, argv, "-f6", 8);
   fused_layers[5] = get_int_arg(argc, argv, "-l6", 16);
   num_devices[5] = get_int_arg(argc, argv, "-d6", 1);
   // sp 7
   partitions_h[6] = get_int_arg(argc, argv, "-n7", 2);
   partitions_w[6] = get_int_arg(argc, argv, "-m7", 2);
   from_layers[6] = get_int_arg(argc, argv, "-f7", 8);
   fused_layers[6] = get_int_arg(argc, argv, "-l7", 16);
   num_devices[6] = get_int_arg(argc, argv, "-d7", 1);
   // sp 8
   partitions_h[7] = get_int_arg(argc, argv, "-n8", 2);
   partitions_w[7] = get_int_arg(argc, argv, "-m8", 2);
   from_layers[7] = get_int_arg(argc, argv, "-f8", 8);
   fused_layers[7] = get_int_arg(argc, argv, "-l8", 16);
   num_devices[7] = get_int_arg(argc, argv, "-d8", 1);
   // sp 9
   partitions_h[8] = get_int_arg(argc, argv, "-n9", 2);
   partitions_w[8] = get_int_arg(argc, argv, "-m9", 2);
   from_layers[8] = get_int_arg(argc, argv, "-f9", 8);
   fused_layers[8] = get_int_arg(argc, argv, "-l9", 16);
   num_devices[8] = get_int_arg(argc, argv, "-d9", 1);
   // sp 10 
   partitions_h[9] = get_int_arg(argc, argv, "-n10", 2);
   partitions_w[9] = get_int_arg(argc, argv, "-m10", 2);
   from_layers[9] = get_int_arg(argc, argv, "-f10", 8);
   fused_layers[9] = get_int_arg(argc, argv, "-l10", 16);
   num_devices[9] = get_int_arg(argc, argv, "-d10", 1);
   // sp 11
   partitions_h[10] = get_int_arg(argc, argv, "-n11", 2);
   partitions_w[10] = get_int_arg(argc, argv, "-m11", 2);
   from_layers[10] = get_int_arg(argc, argv, "-f11", 8);
   fused_layers[10] = get_int_arg(argc, argv, "-l11", 16);
   num_devices[10] = get_int_arg(argc, argv, "-d11", 1);
   // sp 12
   partitions_h[11] = get_int_arg(argc, argv, "-n12", 2);
   partitions_w[11] = get_int_arg(argc, argv, "-m12", 2);
   from_layers[11] = get_int_arg(argc, argv, "-f12", 8);
   fused_layers[11] = get_int_arg(argc, argv, "-l12", 16);
   num_devices[11] = get_int_arg(argc, argv, "-d12", 1);
   // sp 13
   partitions_h[12] = get_int_arg(argc, argv, "-n13", 2);
   partitions_w[12] = get_int_arg(argc, argv, "-m13", 2);
   from_layers[12] = get_int_arg(argc, argv, "-f13", 8);
   fused_layers[12] = get_int_arg(argc, argv, "-l13", 16);
   num_devices[12] = get_int_arg(argc, argv, "-d13", 1);
#if 0
   // sp 14
   partitions_h[13] = get_int_arg(argc, argv, "-n14", 2);
   partitions_w[13] = get_int_arg(argc, argv, "-m14", 2);
   from_layers[13] = get_int_arg(argc, argv, "-f14", 8);
   fused_layers[13] = get_int_arg(argc, argv, "-l14", 16);
   num_devices[13] = get_int_arg(argc, argv, "-d14", 1);
   // sp 15
   partitions_h[14] = get_int_arg(argc, argv, "-n15", 2);
   partitions_w[14] = get_int_arg(argc, argv, "-m15", 2);
   from_layers[14] = get_int_arg(argc, argv, "-f15", 8);
   fused_layers[14] = get_int_arg(argc, argv, "-l15", 16);
   num_devices[14] = get_int_arg(argc, argv, "-d15", 1);
   // sp 16
   partitions_h[15] = get_int_arg(argc, argv, "-n16", 2);
   partitions_w[15] = get_int_arg(argc, argv, "-m16", 2);
   from_layers[15] = get_int_arg(argc, argv, "-f16", 8);
   fused_layers[15] = get_int_arg(argc, argv, "-l16", 16);
   num_devices[15] = get_int_arg(argc, argv, "-d16", 1);
   // sp 17
   partitions_h[16] = get_int_arg(argc, argv, "-n17", 2);
   partitions_w[16] = get_int_arg(argc, argv, "-m17", 2);
   from_layers[16] = get_int_arg(argc, argv, "-f17", 8);
   fused_layers[16] = get_int_arg(argc, argv, "-l17", 16);
   num_devices[16] = get_int_arg(argc, argv, "-d17", 1);
   // sp 18
   partitions_h[17] = get_int_arg(argc, argv, "-n18", 2);
   partitions_w[17] = get_int_arg(argc, argv, "-m18", 2);
   from_layers[17] = get_int_arg(argc, argv, "-f18", 8);
   fused_layers[17] = get_int_arg(argc, argv, "-l18", 16);
   num_devices[17] = get_int_arg(argc, argv, "-d18", 1);
   // sp 19
   partitions_h[18] = get_int_arg(argc, argv, "-n19", 2);
   partitions_w[18] = get_int_arg(argc, argv, "-m19", 2);
   from_layers[18] = get_int_arg(argc, argv, "-f19", 8);
   fused_layers[18] = get_int_arg(argc, argv, "-l19", 16);
   num_devices[18] = get_int_arg(argc, argv, "-d19", 1);
   // sp 20
   partitions_h[19] = get_int_arg(argc, argv, "-n20", 2);
   partitions_w[19] = get_int_arg(argc, argv, "-m20", 2);
   from_layers[19] = get_int_arg(argc, argv, "-f20", 8);
   fused_layers[19] = get_int_arg(argc, argv, "-l20", 16);
   num_devices[19] = get_int_arg(argc, argv, "-d20", 1);
   // sp 21
   partitions_h[20] = get_int_arg(argc, argv, "-n21", 2);
   partitions_w[20] = get_int_arg(argc, argv, "-m21", 2);
   from_layers[20] = get_int_arg(argc, argv, "-f21", 8);
   fused_layers[20] = get_int_arg(argc, argv, "-l21", 16);
   num_devices[20] = get_int_arg(argc, argv, "-d21", 1);
   // sp 22
   partitions_h[21] = get_int_arg(argc, argv, "-n22", 2);
   partitions_w[21] = get_int_arg(argc, argv, "-m22", 2);
   from_layers[21] = get_int_arg(argc, argv, "-f22", 8);
   fused_layers[21] = get_int_arg(argc, argv, "-l22", 16);
   num_devices[21] = get_int_arg(argc, argv, "-d22", 1);
   // sp 23
   partitions_h[22] = get_int_arg(argc, argv, "-n23", 2);
   partitions_w[22] = get_int_arg(argc, argv, "-m23", 2);
   from_layers[22] = get_int_arg(argc, argv, "-f23", 8);
   fused_layers[22] = get_int_arg(argc, argv, "-l23", 16);
   num_devices[22] = get_int_arg(argc, argv, "-d23", 1);
#endif

   uint32_t total_cli_num = get_int_arg(argc, argv, "-total_edge", 1);

   for(uint32_t i=0; i<num_sp; i++) {
    fprintf(stderr, "Split points %lu: [%lu, %lu) w/ %lu x %lu on %lu devices.\n", i,
        from_layers[i], from_layers[i] + fused_layers[i],
        partitions_h[i], partitions_w[i],
        num_devices[i]); 
   }

   // pick a cnn model
   // small
   //char network_file[30] = "models/alexnet.cfg";
   //char weight_file[30] = "models/alexnet.weights";
   // medium
   char network_file[30] = "models/yolo.cfg";
   char weight_file[30] = "models/yolo.weights";
   //
   //char network_file[30] = "models/resnet50.cfg";
   //char weight_file[30] = "models/resnet50.weights";
   //char network_file[30] = "models/resnet152.cfg";
   //char weight_file[30] = "models/resnet152.weights";
   // large
   //char network_file[30] = "models/vgg-16.cfg";
   //char weight_file[30] = "models/vgg-16.weights";

   if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL + 10, TCP, GATEWAY_PUBLIC_ADDR);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "gateway")){
      printf("Gateway device\n");
      printf("We have total %d edge devices now\n", total_cli_num);
      deepthings_gateway(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "data_src")){
      printf("Data source edge device\n");
      printf("This client ID is %d\n", get_int_arg(argc, argv, "-edge_id", 0));
      this_cli_id = get_int_arg(argc, argv, "-edge_id", 0);
      deepthings_victim_edge(num_sp, num_devices, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "non_data_src")){
      printf("Idle edge device\n");
      printf("This client ID is %d\n", get_int_arg(argc, argv, "-edge_id", 0));
      this_cli_id = get_int_arg(argc, argv, "-edge_id", 0);
      deepthings_stealer_edge(num_sp, num_devices, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else if(0 == strcmp(get_string_arg(argc, argv, "-mode", "none"), "estimate")) {
      printf("Find the optimal fused points\n");
      deepthings_estimate(num_sp, partitions_h, partitions_w, from_layers, fused_layers, network_file, weight_file, this_cli_id, total_cli_num, addr_list);
   }else {
      printf("Invalid cmd.\n");
      exit(-1);
   }
   return 0;
}

