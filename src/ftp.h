#ifndef FTP_H
#define FTP_H
#include "configure.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct def_layer_wise_overhead{
  uint32_t opt_dev;
  float time;
  float time_comp;
  float time_comm;
  float bflops;
  float comm_size;
} layer_wise_overhead; 

typedef struct def_ftp_overhead{
  int32_t set;
  uint32_t from_layer;
  uint32_t fused_layers;
  uint32_t opt_dev;
  float time;
  float time_comp;
  float time_comm;
  float bflops;
  float original_bflops; // original comp size to be parallelized (BFLOPS)
  float comm_size;
  float score; // TBD
} ftp_overhead; 

typedef struct partition_range{
    int32_t w1;
    int32_t h1;
    int32_t w2;
    int32_t h2;
    int32_t h;
    int32_t w;
    int32_t c;/*Channel number*/
} tile_region;

typedef struct def_ftp_para{
   int32_t layer_undefined;
   uint32_t partitions;
   uint32_t partitions_w;
   uint32_t partitions_h;
   // can start from any layer, default is 0
   uint32_t from_layer;
   uint32_t fused_layers;
   int gap_after;
   uint32_t task_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
   tile_region input_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   tile_region output_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
} ftp_parameters;

typedef struct def_network_para{
   uint32_t layers;
   uint32_t *stride;
   uint32_t *filter;
   uint32_t *type;
   uint32_t *n; // #filters for cnn layer
   tile_region *input_maps;
   tile_region *output_maps;
} network_parameters;

#if DATA_REUSE
typedef struct def_overlapped_data{
   float *down;
   float *right;
   float *up;
   float *left;
   uint32_t down_size;
   uint32_t right_size;
   uint32_t up_size;
   uint32_t left_size;
   tile_region down_region;
   tile_region right_region;
   tile_region up_region;
   tile_region left_region;
} overlapped_tile_data;

typedef struct def_ftp_parameters_reuse{
   float* shrinked_input[PARTITIONS_MAX];
   uint32_t shrinked_input_size[PARTITIONS_MAX];
   uint32_t adjacent_reuse_data_size[PARTITIONS_MAX];
   uint32_t self_reuse_data_size[PARTITIONS_MAX];
   uint32_t coverage[PARTITIONS_MAX];
   uint32_t missing[PARTITIONS_MAX];
   uint32_t partitions;
   uint32_t partitions_w;
   uint32_t partitions_h;
   uint32_t from_layer;
   uint32_t fused_layers;
   uint32_t task_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
   uint32_t schedule[PARTITIONS_MAX];
   tile_region input_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   tile_region output_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   overlapped_tile_data output_reuse_regions[PARTITIONS_MAX][FUSED_LAYERS_MAX];
} ftp_parameters_reuse;

ftp_parameters_reuse* preform_ftp_reuse(network_parameters* net_para, ftp_parameters* ftp_para);
void partition_and_estimate_reuse(network_parameters* net_para, ftp_parameters* ftp_para, ftp_parameters_reuse* ftp_para_reuse);
uint32_t get_coverage(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id);
void set_coverage(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id);
void set_missing(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id);
uint32_t get_missing(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id);
void clean_coverage(ftp_parameters_reuse* ftp_para_reuse);
bool is_reuse_ready(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id);

tile_region get_region(overlapped_tile_data * overlap, uint32_t pos);
uint32_t get_size(overlapped_tile_data * overlap, uint32_t pos);
float* get_data(overlapped_tile_data * overlap, uint32_t pos);
void set_region(overlapped_tile_data * overlap, uint32_t pos, tile_region tile);
void set_size(overlapped_tile_data * overlap, uint32_t pos, uint32_t size);
void set_data(overlapped_tile_data * overlap, uint32_t pos, float* data);

#endif

ftp_parameters* preform_ftp(uint32_t N, uint32_t M, uint32_t from, uint32_t fused_layers, network_parameters* net_para);
layer_wise_overhead** layer_wise_estimate(network_parameters* net_para);
ftp_overhead* ftp_estimate(network_parameters* net_para, ftp_parameters* ftp_para, layer_wise_overhead** layer_wise_overhead_list);
float dp_buttom_up(uint32_t from_layer, uint32_t fused_layers, network_parameters* net_para, layer_wise_overhead** layer_wise_overhead_list, uint32_t* dp_opt_fused_layers);
void print_tile_region(tile_region tile);
void print_dp_time();
#endif
