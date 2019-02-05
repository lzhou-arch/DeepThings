#include "ftp.h"
#include "configure.h"
#include "inference_engine_helper.h"

#define LOCAL_FACTOR 1.0
#define PARALLEL_OVERHEAD 1.10
#define PARALLEL_GAIN 1.0
#define MAX_LAYERS 48

static inline void grid(network_parameters* net_para, ftp_parameters* ftp_para, uint32_t M, uint32_t N){
   int32_t w = net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w;
   int32_t h = net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h;
   int32_t partition_w = M;
   int32_t partition_h = N;
   int32_t stride_w = ceil(((float)w)/((float)partition_w));    
   int32_t start_w = 0;
   int32_t end_w = stride_w - 1;
   int32_t stride_h = ceil(((float)h)/((float)partition_h));    
   int32_t start_h = 0;
   int32_t end_h = stride_h - 1;
   int32_t i, j, task_id;

   for(i = 0; i < partition_h; i++){
      start_w = 0;
      end_w = stride_w - 1;	 
      for(j = 0; j < partition_w; j++){
         task_id = ftp_para->task_id[i][j];
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w1 = start_w;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w2 = end_w;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h1 = start_h;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h2 = end_h;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h = end_h - start_h + 1;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w = end_w - start_w + 1;
         start_w = end_w + 1;
         if(j == (partition_w-2)) {end_w = w - 1;}
         else {end_w = end_w + stride_w;}
      }
      start_h = end_h + 1;
      if(i == (partition_h-2)) {end_h = h - 1;}
      else {end_h = end_h + stride_h;}
   }
}

/*
Input:
   ftp_para->output_tiles[ftp_para->task_id[i][j]][l]
Output:
   ftp_para->input_tiles[ftp_para->task_id[i][j]][l]; 
*/
static tile_region traversal(network_parameters* net_para, tile_region output, uint32_t l){
   tile_region input; 
   int32_t stride = net_para->stride[l];
   int32_t filter = net_para->filter[l];    
   int32_t w = net_para->input_maps[l].w;
   int32_t h = net_para->input_maps[l].h;     

   if(net_para->type[l] == CONV_LAYER){
      input.w1 = (output.w1*stride - filter/2)>0 ? (output.w1*stride - filter/2) : 0;
      input.w2 = (output.w2*stride + filter/2)<(w-1) ? (output.w2*stride + filter/2) : (w-1);
      input.h1 = (output.h1*stride - filter/2)>0   ? (output.h1*stride - filter/2) : 0;
      input.h2 = (output.h2*stride + filter/2)<(h-1) ? (output.h2*stride + filter/2) : (h-1);
   }else if(net_para->type[l] == POOLING_LAYER){
      input.w1 = output.w1*stride;
      input.w2 = output.w2*stride + stride -1;
      input.h1 = output.h1*stride;
      input.h2 = output.h2*stride + stride -1;
   }else { 
      fprintf(stderr, "Warn: Undefined partition layer: %d\n", l);
   }
   input.w = input.w2 -input.w1 + 1;
   input.h = input.h2 -input.h1 + 1;

#if DEBUG_MULTI_FTP
   fprintf(stderr, " w1 %d, w2 %d, h1 %d, h2 %d, w %d, h %d\n", input.w1, input.w2, input.h1, input.h2, input.w, input.h);
   if (input.w < 0 || input.h < 0) {
     fprintf(stderr, "Error: negative input width/height..\n");
     exit(-1);
   }
#endif
   return input;
}

ftp_parameters* preform_ftp(uint32_t N, uint32_t M, uint32_t from_layer, uint32_t fused_layers, network_parameters* net_para){
#if DEBUG_MULTI_FTP
  fprintf(stderr, "Perform ftp from [%d, %d)\n", from_layer, from_layer + fused_layers);
#endif
  int32_t i, j, l;
  int32_t num_undefined_layers = 0;
  for(l = from_layer+fused_layers-1; l >= (int32_t) from_layer; l--){
   if(net_para->type[l] == CONV_LAYER){
     // 
   }else if(net_para->type[l] == POOLING_LAYER){
     // 
   }else { 
     fprintf(stderr, "Warn: Undefined partition layer: %d\n", l);
     num_undefined_layers++;
   }
  }

   ftp_parameters* ftp_para = (ftp_parameters*)malloc(sizeof(ftp_parameters));
   ftp_para->partitions = N*M;
   ftp_para->partitions_h = N;
   ftp_para->partitions_w = M;
   ftp_para->from_layer = from_layer;
   ftp_para->fused_layers = fused_layers;

   if (num_undefined_layers > 0) {
     ftp_para->layer_undefined = 1;
     return ftp_para;
   }

   //fprintf(stderr, "ftp_para: from_layer %d, fused_layers %d\n", from_layer, fused_layers);

   int32_t id = 0;
   for(i = 0; i < ftp_para->partitions_h; i++){
      for(j = 0; j < ftp_para->partitions_w; j++){
         ftp_para->task_id[i][j] = id;
         id++;
      }
   }
   grid(net_para, ftp_para, M, N);
   for(i = 0; i < ftp_para->partitions_h; i++){
      for(j = 0; j < ftp_para->partitions_w; j++){
#if DEBUG_MULTI_FTP
        fprintf(stderr, "FTP for task %d:\n", ftp_para->task_id[i][j]);
#endif
         // offset: from_layer
         for(l = from_layer+fused_layers-1; l >= (int32_t) from_layer; l--){
#if DEBUG_MULTI_FTP
            fprintf(stderr, " layer %d:\t", l);
#endif
            ftp_para->input_tiles[ftp_para->task_id[i][j]][l-from_layer] = 
                       traversal(net_para, ftp_para->output_tiles[ftp_para->task_id[i][j]][l-from_layer], l);
            if(l>from_layer) ftp_para->output_tiles[ftp_para->task_id[i][j]][l-from_layer-1] 
                     = ftp_para->input_tiles[ftp_para->task_id[i][j]][l-from_layer];
         }
      }
   }
   return ftp_para;
}

layer_wise_overhead** layer_wise_estimate(network_parameters* net_para){
   int32_t l;

   float comp_size; // bflops
   float comm_size; // KB
   float comm_size_layer_wise; // KB

   // conv var1 input, var2 stride
   // maxpool var1 input, var2 output
   float comp_var1, comp_var2;

   // linear regression model
   //float coef_conv_inter = 0.04641, coef_conv_inputs = 0.0000009342, coef_conv_filters = 0.0001828;
   float coef_conv_inter = 0.1049, coef_conv_inputs = 0.0000008736, coef_conv_filters = 0.0004246; // vgg-16 only
   //float coef_conv_inter = -0.03344, coef_conv_inputs = 0.0000006455, coef_conv_filters = 0.0001559; // yolov2 only
   //float coef_conv_inter = -0.0518, coef_conv_inputs = 0.000001101, coef_conv_filters = 0.00006975; // alexnet only
   float coef_maxpool_inter = 0.0354664, coef_maxpool_inputs = -0.0002746, coef_maxpool_outputs = 0.0011031;

   float tx1 = 0.000361*10, tx2 = 0.0983*10; // slowest x10
   //float tx1 = 0.000361, tx2 = 0.0983; // my wifi
   //float tx1 = 0.0002, tx2 = 0.002; // iros paper
   //float tx1 = 0.00002, tx2 = 0.0002; // 1/10 
   //float tx1 = 0.000000000001, tx2 = 0.000000000001; // simulate local memory

   float billion = 1000000000.;
   
   float total_time, total_comp_time, total_comm_time;
   float total_comp, total_comm;
   float total_time_1d = 0;

   layer_wise_overhead** layer_wise_overhead_list = (layer_wise_overhead**)malloc(sizeof(layer_wise_overhead*)*net_para->layers);

   // calculate computation for layer-wise in BFLOPs per layer
   // count communication for layer-wise and fused-layer in KB
   for(l = 0; l < net_para->layers; l++) {
     int32_t n = net_para->n[l];
     int32_t size = net_para->filter[l];
     int32_t c1 = net_para->input_maps[l].c;
     int32_t stride = net_para->stride[l];

     int32_t layer_supported = 1;

     //TODO(lizhou): add the overlap per layer for different parallelism config e.g. height/width/channel

     // linear for layer-wise
     if (net_para->type[l] == CONVOLUTIONAL) {
       comp_var1 = c1*net_para->input_maps[l].w*net_para->input_maps[l].h;
       comp_var2 = (float)(size*size)/(float)(stride*stride)*n;
       comp_size = (2.0*n*size*size*c1)*(net_para->output_maps[l].w*net_para->output_maps[l].h)/billion;
     } else if (net_para->type[l] == MAXPOOL) {
       comp_var1 = net_para->input_maps[l].w*net_para->input_maps[l].h;
       comp_var2 = net_para->output_maps[l].w*net_para->output_maps[l].h;
       comp_size = (size*size*c1)*(net_para->output_maps[l].w*net_para->output_maps[l].h)/billion;
     } else {
       layer_supported = 0;
     }

     // find layer-wise optimal #dev
     // d = 1, t = tc; d > 1, t = tc + tx
     uint32_t opt_d = 1;
     float t_layer_min, t_layer_1d, t_layer;
     float t_layer_comp, t_layer_comm;

     // TODO(lizhou): add fake estimation for other types of layers.
     if (layer_supported == 0) {
       fprintf(stderr, "Warning: unsupported layer.\n");
       // only supported parallelize conv and maxpool, process other layers locally
       // no comm cost, comp is estimated by a fixed value.
       //t_layer_min = t_layer_min = t_layer_1d = 0.5; // too high for alexnet 
       //t_layer_comp = t_layer_min = t_layer_1d = 0.05; // too high for alexnet 
       t_layer_comp = t_layer_min = t_layer_1d = 0.5; // too high for alexnet 
       t_layer_comm = 0; 
       opt_d = 1; 
       comp_size = 0.05;
       comm_size_layer_wise = 0;
       if (net_para->type[l] == SHORTCUT) {
         t_layer_comp = t_layer_min = t_layer_1d = 0;
         t_layer_comm = 0; 
         opt_d = 1; 
         comp_size = 0; // output size? 
         comm_size_layer_wise = 0;
       }
     } else {
       // parallelize conv and maxpool
       // layer-wise comm, includes input and output tensor
       uint32_t comm_size_in = sizeof(float)*(net_para->input_maps[l].w*net_para->input_maps[l].h*net_para->input_maps[l].c); 
       //uint32_t comm_size_out = sizeof(float)*(net_para->output_maps[l].w*net_para->output_maps[l].h*net_para->output_maps[l].c);
       // assume that output and input is overlapped for each layer
       uint32_t comm_size_out = 0;
       comm_size = comm_size_layer_wise = (float)(comm_size_in+comm_size_out)/1024.;
       printf("Info layer-wise(%u): %f KB\n", l, comm_size);

       // 1 device has no comm cost
       if (net_para->type[l] == CONVOLUTIONAL) {
         t_layer_1d = coef_conv_inter + coef_conv_inputs*comp_var1 + coef_conv_filters*comp_var2;
       } else if (net_para->type[l] == MAXPOOL) {
         t_layer_1d = coef_maxpool_inter + coef_maxpool_inputs*comp_var1 + coef_maxpool_outputs*comp_var2;
       } else {
         fprintf(stderr, "Error: unsupported layer.\n");
         exit(-1);
       }
       printf("DEBUG layer-wise(%u) 1 devices: tc %f (%f BFLOPS, %f KB: %f + %f)\n", l, t_layer_1d, comp_size, comm_size, (float)comm_size_in/1024., (float)comm_size_out/1024.);
       t_layer_min = t_layer_1d;
       t_layer_comp = t_layer_1d;
       t_layer_comm = 0;

       for (int d = 2; d <= MAX_EDGE_NUM; d++) {
         // consider the parallel coef, not exactly divide by d, with 10% overhead
         float tc = t_layer_1d/d*PARALLEL_OVERHEAD; // decrease
         // Note: since comm and comp are pipelined, tx = (d-1) tx(in) + 1 tx(out)
         // tx can take more time, if tx(out) > tx(in), should add wait time
         // uint32_t diff_comm_size = (comm_size_out > comm_size_in) ? comm_size_out - comm_size_in : 0;
         uint32_t num_partitions = d;
         // layer-wise comm, includes input and output tensor, should also decrease
         float tx = (tx1*comm_size_in*(float)(d-LOCAL_FACTOR)/(d*1024.)+tx2*num_partitions*(float)(d-LOCAL_FACTOR)/(float)(d));
           //+ (tx1*comm_size_out*(float)(d-LOCAL_FACTOR)/(d*(d-1)*1024.)+tx2);
           //+ (diff_comm_size > 0) ? (tx1*diff_comm_size*(float)(d-LOCAL_FACTOR)/(d*(d-1)*1024.)+tx2) : 0;
         t_layer = tc + tx;
         printf("INFOTIME layer-wise(%u) %d devices: tc vs tx: %f, %f, total: %f\n", l, d, tc, tx, t_layer);
         if (t_layer*PARALLEL_GAIN < t_layer_min) {
           printf("Add Dev (%d -> %d)\n", opt_d, d);
           t_layer_min = t_layer;
           t_layer_comp = tc;
           t_layer_comm = tx;
           opt_d = d;
           comm_size_layer_wise = comm_size*(float)(d-LOCAL_FACTOR)/(float)(d);
         }
       }
     }

     //  record 1d performance as baseline
     total_time_1d += t_layer_1d;

     layer_wise_overhead_list[l] = (layer_wise_overhead*)malloc(sizeof(layer_wise_overhead));
     layer_wise_overhead_list[l]->time = t_layer_min; 
     layer_wise_overhead_list[l]->time_comp = t_layer_comp; 
     layer_wise_overhead_list[l]->time_comm = t_layer_comm; 
     layer_wise_overhead_list[l]->opt_dev = opt_d; 
     layer_wise_overhead_list[l]->bflops = comp_size;
     layer_wise_overhead_list[l]->comm_size = comm_size_layer_wise;

     printf("Layer %u opt_dev: %d, layer-wise time(s): %f vs. %f (%fKB) accum 1d: %f\n", l, opt_d, t_layer_min, t_layer_1d, comm_size_layer_wise, total_time_1d);
   } // end for

   total_time = total_comp_time = total_comm_time = total_comp = total_comm = 0;
   for(l = 0; l < net_para->layers; l++) {
     layer_wise_overhead* lwo = layer_wise_overhead_list[l];
     total_time += lwo->time;
     total_comp_time += lwo->time_comp;
     total_comm_time += lwo->time_comm;
     total_comp += lwo->bflops;
     total_comm += lwo->comm_size;
     printf("Summary layer-wise layer %u (opt_dev: %d), time: %f, comp: %f, (%f BFLOPs), comm: %f (%f KB)\n", l, lwo->opt_dev, lwo->time, lwo->time_comp, lwo->bflops, lwo->time_comm, lwo->comm_size);
   }
   printf("Summary all layer-wise layers, time: %f (vs. 1d %f), comp: %f (%f BFLOPs), comm: %f (%f KB)\n", total_time, total_time_1d, total_comp_time, total_comp, total_comm_time, total_comm);
   return layer_wise_overhead_list;
}

// store the overhead for layer fusion [from_layer, from_layer+fused_layers)
static ftp_overhead* fused_layer_overhead_list[MAX_LAYERS][MAX_LAYERS];
static int32_t ftp_overhead_map[MAX_LAYERS][MAX_LAYERS];
// TODO(lizhou): use struct
static float dp[MAX_LAYERS][MAX_LAYERS]; // store the dp result
static float dp_tx[MAX_LAYERS][MAX_LAYERS]; // store the dp result
static float dp_tc[MAX_LAYERS][MAX_LAYERS]; // store the dp result

ftp_overhead* ftp_estimate(network_parameters* net_para, ftp_parameters* ftp_para, layer_wise_overhead** layer_wise_overhead_list){
   uint32_t task;
   uint32_t data_size;
   uint32_t input_size = 0;
   uint32_t output_size = 0;
   uint32_t dw1, dw2;
   uint32_t dh1, dh2;
   uint32_t i, j;
   int32_t l;

   printf("Task size:\n");
   for(i = 0; i < ftp_para->partitions_h; i++){
      for(j = 0; j < ftp_para->partitions_w; j++){
         task = ftp_para->task_id[i][j];
         dw1 = ftp_para->input_tiles[task][0].w1;
         dw2 = ftp_para->input_tiles[task][0].w2;
         dh1 = ftp_para->input_tiles[task][0].h1;
         dh2 = ftp_para->input_tiles[task][0].h2;
         data_size = sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->input_maps[ftp_para->from_layer].c;
         input_size += data_size;
         printf("Variation fused-layer [%d,%d) task size: %u\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, data_size); 
         dw1 = ftp_para->output_tiles[task][ftp_para->fused_layers-1].w1;
         dw2 = ftp_para->output_tiles[task][ftp_para->fused_layers-1].w2;
         dh1 = ftp_para->output_tiles[task][ftp_para->fused_layers-1].h1;
         dh2 = ftp_para->output_tiles[task][ftp_para->fused_layers-1].h2;
         data_size = sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c;
         output_size += data_size;
         printf("Variation fused-layer [%d,%d) output size: %u\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, data_size); 
      }
   }
   printf("\n");
   data_size = sizeof(float)*net_para->input_maps[ftp_para->from_layer].w*net_para->input_maps[ftp_para->from_layer].h*net_para->input_maps[ftp_para->from_layer].c;
   printf("Fused_layers [%d, %d) Input data_size vs. original_data_size: %u/%u (+%f%)\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, input_size, data_size, (float)(input_size-data_size)/data_size); 

   float comp_size; // bflops
   float ck_comp_size; // bflops
   float comp_size_fused_layer;
   float total_comp_size = 0;
   float original_total_comp_size = 0;

   float comm_size;  // KB
   float comm_size_fused_layer; // KB
   float total_comm_size_layer_wise = 0; // KB

   // conv var1 input, var2 stride
   // maxpool var1 input, var2 output
   float comp_conv_var1, comp_conv_var2; // fused_layer
   float comp_maxpool_var1, comp_maxpool_var2; // fused_layer
   comp_conv_var1 = comp_conv_var2 = comp_maxpool_var1 = comp_maxpool_var2 = 0;

   float ck_comp_conv_var1, ck_comp_conv_var2; // fused_layer
   float ck_comp_maxpool_var1, ck_comp_maxpool_var2; // fused_layer
   ck_comp_conv_var1 = ck_comp_conv_var2 = ck_comp_maxpool_var1 = ck_comp_maxpool_var2 = 0;

   float total_time_fused_layer;
   float total_time_layer_wise = 0;

   float billion = 1000000000.;

   // linear regression model
   // TODO(lizhou): validate the coef value under ftp partitions
   //float coef_conv_inter = 0.04641, coef_conv_inputs = 0.0000009342, coef_conv_filters = 0.0001828;
   float coef_conv_inter = 0.1049, coef_conv_inputs = 0.0000008736, coef_conv_filters = 0.0004246; // vgg-16 only
   //float coef_conv_inter = -0.03344, coef_conv_inputs = 0.0000006455, coef_conv_filters = 0.0001559; // yolov2 only
   //float coef_conv_inter = -0.0518, coef_conv_inputs = 0.000001101, coef_conv_filters = 0.00006975; // alexnet only
   float coef_maxpool_inter = 0.0354664, coef_maxpool_inputs = -0.0002746, coef_maxpool_outputs = 0.0011031;

   float tx1 = 0.000361*10, tx2 = 0.0983*10; // slowest x10
   //float tx1 = 0.000361, tx2 = 0.0983; // tx1 * x + tx2 my wifi
   //float tx1 = 0.0002, tx2 = 0.002; // iros paper
   //float tx1 = 0.00002, tx2 = 0.0002; // 1/10
   //float tx1 = 0.000000000001, tx2 = 0.000000000001; // simulate local memory

   // number of conv in fused_layer
   int32_t num_conv = 0;

   // layer wise time and count num of conv layers
   for(l = ftp_para->from_layer+ftp_para->fused_layers-1; l >= (int32_t)ftp_para->from_layer; l--) {
     total_time_layer_wise += layer_wise_overhead_list[l]->time;
     total_comm_size_layer_wise += layer_wise_overhead_list[l]->comm_size;
     if (net_para->type[l+ftp_para->from_layer] == CONVOLUTIONAL) {
       num_conv++;
     }
   }
   printf("Layer [%u~%u) min layer-wise time: %f\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, total_time_layer_wise);

   float* comp_size_partition = (float*)malloc(sizeof(float)*ftp_para->partitions_h*ftp_para->partitions_w);

   for(i = 0; i < ftp_para->partitions_h; i++){
     for(j = 0; j < ftp_para->partitions_w; j++){
       task = ftp_para->task_id[i][j];
       comp_size_partition[task] = 0;
     }
   }

   // calculate computation for fused-layer in BFLOPs per layer
   // count communication for layer-wise and fused-layer in KB
   for(l = ftp_para->fused_layers-1; l >= 0; l--) {
     comp_size_fused_layer = 0;
     int32_t n = net_para->n[l+ftp_para->from_layer];
     int32_t size = net_para->filter[l+ftp_para->from_layer];
     int32_t c1 = net_para->input_maps[l+ftp_para->from_layer].c;
     int32_t stride = net_para->stride[l+ftp_para->from_layer];

     // check 
     if (net_para->type[l+ftp_para->from_layer] == CONVOLUTIONAL) {
       ck_comp_conv_var1 = c1*net_para->input_maps[l+ftp_para->from_layer].w*net_para->input_maps[l+ftp_para->from_layer].h;
       ck_comp_conv_var2 = (float)(size*size)/(float)(stride*stride)*n;
       ck_comp_size = (2.0*n*size*size*c1)*(net_para->output_maps[l+ftp_para->from_layer].w*net_para->output_maps[l+ftp_para->from_layer].h)/billion;
     } else if (net_para->type[l+ftp_para->from_layer] == MAXPOOL) {
       ck_comp_maxpool_var1 = net_para->input_maps[l+ftp_para->from_layer].w*net_para->input_maps[l+ftp_para->from_layer].h;
       ck_comp_maxpool_var2 = net_para->output_maps[l+ftp_para->from_layer+ftp_para->fused_layers].w*net_para->output_maps[l+ftp_para->from_layer+ftp_para->fused_layers].h;
       ck_comp_size = (size*size*c1)*(net_para->output_maps[l+ftp_para->from_layer].w*net_para->output_maps[l+ftp_para->from_layer].h)/billion;
     } 

     for(i = 0; i < ftp_para->partitions_h; i++){
       for(j = 0; j < ftp_para->partitions_w; j++){
         task = ftp_para->task_id[i][j];
         // accumulate total conv and maxpool comp 
         if (net_para->type[l+ftp_para->from_layer] == CONVOLUTIONAL) {
           comp_conv_var1 += c1*ftp_para->input_tiles[task][l].w*ftp_para->input_tiles[task][l].h;
           // TODO(lizhou): adjust the model for ftp, now add once for same layer 
           // NOTE: may lead to smaller results for 1 device.
           comp_conv_var2 = (float)(size*size)/(float)(stride*stride)*n;
           comp_size = ((2.0*n*size*size*c1)*(ftp_para->output_tiles[task][l].w*ftp_para->output_tiles[task][l].h)/billion); // BFLOPS
         } else if (net_para->type[l+ftp_para->from_layer] == MAXPOOL) {
           comp_maxpool_var1 += ftp_para->input_tiles[task][l].w*ftp_para->input_tiles[task][l].h;
           comp_maxpool_var2 += ftp_para->output_tiles[task][l].w*ftp_para->output_tiles[task][l].h;
           comp_size = ((size*size*c1)*(ftp_para->output_tiles[task][l].w*ftp_para->output_tiles[task][l].h)/billion);
         } else {
           fprintf(stderr, "Error: unsupported layer\n");
           exit(-1);
         }
         comp_size_fused_layer += comp_size;
         comp_size_partition[task] += comp_size;
       }
     }
     // original comp in BFLOPS
     if (net_para->type[l+ftp_para->from_layer] == CONVOLUTIONAL) {
       comp_size = ((2.0*n*size*size*c1)*(net_para->output_maps[l+ftp_para->from_layer].w*net_para->output_maps[l+ftp_para->from_layer].h)/billion);
     } else if (net_para->type[l+ftp_para->from_layer] == MAXPOOL) {
       comp_size = ((size*size*c1)*(net_para->output_maps[l+ftp_para->from_layer].w*net_para->output_maps[l+ftp_para->from_layer].h)/billion);
     } else {
       exit(-1);
     }
     original_total_comp_size += comp_size; 
     total_comp_size += comp_size_fused_layer; 
     printf("Layer %u comp_size_fused_layer vs. comp_size_original: %f/%f (+%f%) (BFLOPs)\n", l+ftp_para->from_layer, comp_size_fused_layer, comp_size, (comp_size_fused_layer-comp_size)/(comp_size+0.000001)*100.); 
   } // end for

   // check 
   printf("CKKKK conv (%f, %f; %f, %f), maxpool (%f, %f: %f, %f)\n", comp_conv_var1, ck_comp_conv_var1, comp_conv_var2, ck_comp_conv_var2, comp_maxpool_var1, ck_comp_maxpool_var1, comp_maxpool_var2, ck_comp_maxpool_var2);

   // assume that workload is evenly distributed, comm includes input and output layer
   // input layer contains overlap
   uint32_t comm_size_in = 0;
   for(i = 0; i < ftp_para->partitions_h; i++){
     for(j = 0; j < ftp_para->partitions_w; j++){
       task = ftp_para->task_id[i][j];
       comm_size_in += sizeof(float)*ftp_para->input_tiles[task][0].w*ftp_para->input_tiles[task][0].h*net_para->input_maps[ftp_para->from_layer].c;
       printf("Variation fused-layer [%d,%d) partition %d comp size: %f\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, task, comp_size_partition[task]);
     }
   }
   uint32_t comm_size_out = sizeof(float)*(net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w*net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h*net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c);
   comm_size = comm_size_fused_layer = (comm_size_in+comm_size_out)/1024.;

   // cal fused_layers exec time
   int opt_d = 1;
   float t_layer_comp, t_layer_comm;
   float t_fused_layer;
   float total_time_fused_layer_1d = coef_conv_inter // num_conv*coef_conv_inter
     + coef_conv_inputs*comp_conv_var1 + coef_conv_filters*comp_conv_var2
     + coef_maxpool_inter // (ftp_para->fused_layers-num_conv)*coef_maxpool_inter
     + coef_maxpool_inputs*comp_maxpool_var1 + coef_maxpool_outputs*comp_maxpool_var2;
   // increase 1d overhead only!
   total_time_fused_layer_1d *= PARALLEL_OVERHEAD;
   printf("DEBUG fused-layer [%u, %u) 1 devices: tc %f (%f BFLOPS, %f KB: %f KB + %f KB)\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, total_time_fused_layer_1d, total_comp_size, comm_size, (float)comm_size_in/1024., (float)comm_size_out/1024.);
   total_time_fused_layer = total_time_fused_layer_1d; 
   t_layer_comp = total_time_fused_layer_1d;
   t_layer_comm = 0;

   for (int d = 2; d <= MAX_EDGE_NUM; d++) {
     // add penalty
     float tc = total_time_fused_layer_1d/d*PARALLEL_OVERHEAD;
     // TODO(lizhou): assume the partition is exactly the same with number of devices
     // for this equation but should be updated to par_h*par_w
     uint32_t diff_comm_size = (comm_size_out > comm_size_in) ? comm_size_out - comm_size_in : 0;
     //int32_t num_partitions = d;
     uint32_t num_partitions = ftp_para->partitions_h*ftp_para->partitions_w; // use this will increase tc, since tx cost is higher
     // layer-wise comm, includes input and output tensor, should also decrease
     //float tx = (tx1*comm_size_in*(float)(d-LOCAL_FACTOR)/(d*1024.)+tx2*num_partitions*(float)(d-LOCAL_FACTOR)/(float)d)
     float tx = (tx1*comm_size_in*(float)(d-LOCAL_FACTOR)/(d*1024.)+tx2*d*(float)(d-LOCAL_FACTOR)/(float)d)
       + (tx1*comm_size_out*(float)(d-LOCAL_FACTOR)/(d*(d-1)*1024.)+tx2);
       // Warn: Use this with small number of partitions
       //+ (diff_comm_size > 0) ? (tx1*diff_comm_size*(float)(d-LOCAL_FACTOR)/(d*(d-1)*1024.)+tx2) : 0;

     float t_wait = 0;
     // Warn: if num_partitions >> d, no more devices are available for continous computation, and need to wait.
     // TODO(lizhou): check!!
     //float tin = (tx1*comm_size_in*(float)(d-LOCAL_FACTOR)/(d*num_partitions*1024.)+tx2*(float)(d-LOCAL_FACTOR)/(float)d);
     //int32_t counter = 0;
     //if (tin*(d-1)*1.2 < tc) {
     //  counter = (int32_t)(num_partitions/d)-1 + (num_partitions%d > 0 ? 1 : 0); 
     //  t_wait = (float)counter*(tc-tin*(d-1));
     //  printf("Warn!!: add more devices for %d partitions (%d), tc vs (d-1)*tin, %f, %f, overhead (%d): %f\n", num_partitions, d, tc*(d-1), tin, counter, t_wait);
     //}

     t_fused_layer = tc + tx;
     t_fused_layer += t_wait;

     printf("DEBUG fused-layer [%u, %u) %d devices: tc vs tx: %f, %f, t_wait: %f, total: %f\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, d, tc, tx, t_wait, t_fused_layer);
     printf("Info fused-layer [%u~%u) %d devices, comm_size_fused_layer vs. comm_size_layer_wise: %f/%f (KB) (-%f%) comm\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, d, comm_size*(float)(d-LOCAL_FACTOR)/d, total_comm_size_layer_wise, (total_comm_size_layer_wise - comm_size*(float)(d-LOCAL_FACTOR)/d)/total_comm_size_layer_wise*100.); 
     if (t_fused_layer*PARALLEL_GAIN < total_time_fused_layer) {
       total_time_fused_layer = t_fused_layer;
       t_layer_comp = tc;
       t_layer_comm = tx;
       opt_d = d;
       comm_size_fused_layer = comm_size*(float)(d-LOCAL_FACTOR)/d;
     }
   }

   ftp_overhead* ftp_o = (ftp_overhead*)malloc(sizeof(ftp_overhead));
   ftp_o->time = total_time_fused_layer; 
   ftp_o->time_comp = t_layer_comp; 
   ftp_o->time_comm = t_layer_comm; 
   ftp_o->opt_dev = opt_d; 
   ftp_o->bflops = total_comp_size;
   ftp_o->comm_size = comm_size_fused_layer;

   printf("Layer [%u~%u), comp_size_fused_layer vs. comp_size_original: %f/%f (+%f%) (BFLOPs), comm_size_fused_layer vs. comm_size_layer_wise: %f/%f (KB) (-%f%) comm\n", ftp_para->from_layer, ftp_para->from_layer+ftp_para->fused_layers, total_comp_size, original_total_comp_size, (total_comp_size-original_total_comp_size)/original_total_comp_size*100., comm_size_fused_layer, total_comm_size_layer_wise, (total_comm_size_layer_wise - comm_size_fused_layer)/total_comm_size_layer_wise*100.); 
   printf("TIME DEBUG fused-layers(#dev) vs. layer-wise: %f(%d) vs. %f\n", total_time_fused_layer, opt_d, total_time_layer_wise);

   // evaluation score
   //ftp_o->score = (float)original_total_comp_size/(ftp_o->extra_comp_size+(float)total_comm_size*comm_comp_ratio+1);
   
   return ftp_o;
}

// DP: rn = min_{1<=i<=n} (pi+rn-i)
// Return opt sol time for layers [from_layer, from_layer+fused_layers)
// Sol is saved in dy[from_layer][fused_layers]
float dp_buttom_up(uint32_t from_layer, uint32_t fused_layers, network_parameters* net_para, layer_wise_overhead** layer_wise_overhead_list, uint32_t* dp_opt_fused_layers) {
   int32_t i, j;
   float min_total_time = 1000000;
   int32_t opt_fused_layers = 0;

   if (dp[from_layer][fused_layers] > 0.000001)
     return dp[from_layer][fused_layers];
   
   if (fused_layers < 1) return 0;

   printf("Target: estimate opt cost for layers [%d, %d) ...\n", from_layer, from_layer+fused_layers);
   for(i = 1; i <= (int32_t)fused_layers; i++) {
     float total_time;
     // when i = 1, indicates only a single layer 
     if(i == 1) {
       total_time = layer_wise_overhead_list[from_layer]->time + dp_buttom_up(from_layer+i, fused_layers-i, net_para, layer_wise_overhead_list, dp_opt_fused_layers);
     } else {
       //printf("Try fuse layers [%d, %d) ...\n", from_layer, from_layer+i);
       // 2x2 by default
       uint32_t N, M;
       N = M = 4; 

       float time_fused_layer;

       if (ftp_overhead_map[from_layer][i] == 1) {
         time_fused_layer = fused_layer_overhead_list[from_layer][i]->time;
       } else {
         ftp_parameters* ftp_para = (ftp_parameters*)malloc(sizeof(ftp_parameters));
         ftp_para = preform_ftp(N, M, from_layer, i, net_para);
         if (ftp_para->layer_undefined == 1) {
           // range of layers can't be fused, use layer-wise results instead
           for(j = from_layer; j < from_layer+i; j++) {
             time_fused_layer += layer_wise_overhead_list[j]->time;
           }
           fused_layer_overhead_list[from_layer][i] = (ftp_overhead*)malloc(sizeof(ftp_overhead));
           fused_layer_overhead_list[from_layer][i]->time = time_fused_layer;
         } else {
           // ftp overhead  for [from_layer, from_layer+i)
           fused_layer_overhead_list[ftp_para->from_layer][ftp_para->fused_layers] =
             ftp_estimate(net_para, ftp_para, layer_wise_overhead_list);
           time_fused_layer = fused_layer_overhead_list[ftp_para->from_layer][ftp_para->fused_layers]->time;
         }
         ftp_overhead_map[from_layer][i] = 1;
       }

       total_time = time_fused_layer + dp_buttom_up(from_layer+i, fused_layers-i, net_para, layer_wise_overhead_list, dp_opt_fused_layers);
     }
     if (total_time < min_total_time) {
       min_total_time = total_time;
       opt_fused_layers = i;
       //printf(" TEMP OPT fuse [%d, %d)\n", from_layer, from_layer+i);
     }
   }

   dp[from_layer][fused_layers] = min_total_time;
   dp_opt_fused_layers[from_layer] = opt_fused_layers;
   printf("DEBUG OPT FUSED-LAYERS [%u, %u) (%d layers) \n", from_layer, from_layer+opt_fused_layers, opt_fused_layers);
   return min_total_time;
}

static float dp_time_comp, dp_time_comm;
void print_dp_time(layer_wise_overhead** layer_wise_overhead_list, uint32_t from_layer, uint32_t fused_layers, int32_t end){
  if (end == 1) printf("Total OPT time: %f (tc: %f, tx: %f)\n", dp_time_comp+dp_time_comm, dp_time_comp, dp_time_comm);
  else {
    float tmp_time_comp, tmp_time_comm;
    tmp_time_comp = tmp_time_comm = 0;
    if (ftp_overhead_map[from_layer][fused_layers] == 1) {
      tmp_time_comp = fused_layer_overhead_list[from_layer][fused_layers]->time_comp;
      tmp_time_comm = fused_layer_overhead_list[from_layer][fused_layers]->time_comm;
    } else {
      for (int32_t l = from_layer; l < (int32_t)from_layer+fused_layers; l++) {
        tmp_time_comp += layer_wise_overhead_list[l]->time_comp;
        tmp_time_comm += layer_wise_overhead_list[l]->time_comm;
      }
    }
    dp_time_comp += tmp_time_comp;
    dp_time_comm += tmp_time_comm;
    printf("time: %f (tc: %f, tx: %f)\n", tmp_time_comp+tmp_time_comm, tmp_time_comp, tmp_time_comm);
  }
}

#if DATA_REUSE
/*Establish a dependency list, 0 means no dependencies, 1 depends on 0, 2 depends on 1 ...*/
/*For current implementation, we only have 2 levels of dependency*/
/*For example, in a 3x3 grid, the dependency is like below:       
|_0_|_1_|_0_|
|_1_|_0_|_1_|
|_0_|_1_|_0_|
, where tiles with dependency level 1 will need the overlapped data generated by tiles with level 0
*/
void reuse_aware_schedule(ftp_parameters_reuse* ftp_para_reuse){
   int32_t i, j;
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = (i) % 2; j < ftp_para_reuse->partitions_w; j = j + 2){ 
         ftp_para_reuse->schedule[ftp_para_reuse->task_id[i][j]] = 0;
      }
   }
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = (i + 1) % 2; j < ftp_para_reuse->partitions_w; j = j + 2){ 
         ftp_para_reuse->schedule[ftp_para_reuse->task_id[i][j]] = 1;
      }
   }
}

tile_region remove_and_record_overlapped_region_at_output(uint32_t i, uint32_t j,  uint32_t l, 
                                                     ftp_parameters_reuse* ftp_para_reuse, tile_region all_region){
   int adjacent_task;
   tile_region remaining_region = all_region;
   /*Processing the block on the left*/
   overlapped_tile_data overlapped_region;
   if(j > 0) {
      adjacent_task = ftp_para_reuse->task_id[i][j-1]; 
      remaining_region.w1 = ftp_para_reuse->output_tiles[adjacent_task][l].w2 + 1;

      overlapped_region = ftp_para_reuse->output_reuse_regions[adjacent_task][l];
      overlapped_region.right_region.w1 = all_region.w1;
      overlapped_region.right_region.w2 = ftp_para_reuse->output_tiles[adjacent_task][l].w2;
      overlapped_region.right_region.h1 = all_region.h1;
      overlapped_region.right_region.h2 = all_region.h2;
      overlapped_region.right_region.w = overlapped_region.right_region.w2 - overlapped_region.right_region.w1 + 1;
      overlapped_region.right_region.h = overlapped_region.right_region.h2 - overlapped_region.right_region.h1 + 1;
      ftp_para_reuse->output_reuse_regions[adjacent_task][l] = overlapped_region;
#if DEBUG_FTP
      //printf("---(layer %3d), left---\n", l);
      printf("---(layer %3d), left---\n", l+ftp_para_reuse+1);
      print_tile_region(overlapped_region.right_region);
#endif
   }
   /*Processing the block above*/
   if(i > 0) {
      adjacent_task = ftp_para_reuse->task_id[i-1][j]; 
      remaining_region.h1 = ftp_para_reuse->output_tiles[adjacent_task][l].h2 + 1;

      overlapped_region = ftp_para_reuse->output_reuse_regions[adjacent_task][l];
      overlapped_region.down_region.w1 = all_region.w1;
      overlapped_region.down_region.w2 = all_region.w2;
      overlapped_region.down_region.h1 = all_region.h1;
      overlapped_region.down_region.h2 = ftp_para_reuse->output_tiles[adjacent_task][l].h2;
      overlapped_region.down_region.w = overlapped_region.down_region.w2 - overlapped_region.down_region.w1 + 1;
      overlapped_region.down_region.h = overlapped_region.down_region.h2 - overlapped_region.down_region.h1 + 1;
      ftp_para_reuse->output_reuse_regions[adjacent_task][l] = overlapped_region;
#if DEBUG_FTP
      //printf("---(layer %3d), above---\n", l);
      printf("---(layer %3d), above---\n", l+ftp_para_reuse+1);
      print_tile_region(overlapped_region.down_region);
#endif
   }
   /*Processing the block on the right*/
   if((j + 1) < ftp_para_reuse->partitions_w) {
      adjacent_task = ftp_para_reuse->task_id[i][j+1]; 
      remaining_region.w2 = ftp_para_reuse->output_tiles[adjacent_task][l].w1 - 1;

      overlapped_region = ftp_para_reuse->output_reuse_regions[adjacent_task][l];
      overlapped_region.left_region.w1 = ftp_para_reuse->output_tiles[adjacent_task][l].w1;
      overlapped_region.left_region.w2 = all_region.w2;
      overlapped_region.left_region.h1 = all_region.h1;
      overlapped_region.left_region.h2 = all_region.h2;
      overlapped_region.left_region.w = overlapped_region.left_region.w2 - overlapped_region.left_region.w1 + 1;
      overlapped_region.left_region.h = overlapped_region.left_region.h2 - overlapped_region.left_region.h1 + 1;
      ftp_para_reuse->output_reuse_regions[adjacent_task][l] = overlapped_region;
#if DEBUG_FTP
      //printf("---(layer %3d), right---\n", l);
      printf("---(layer %3d), right---\n", l+ftp_para_reuse+1);
      print_tile_region(overlapped_region.left_region);
#endif
   }
   /*Processing the block below*/
   if((i + 1) < ftp_para_reuse->partitions_h) {
      adjacent_task = ftp_para_reuse->task_id[i+1][j]; 
      remaining_region.h2 = ftp_para_reuse->output_tiles[adjacent_task][l].h1 - 1;

      overlapped_region = ftp_para_reuse->output_reuse_regions[adjacent_task][l];
      overlapped_region.up_region.w1 = all_region.w1;
      overlapped_region.up_region.w2 = all_region.w2;
      overlapped_region.up_region.h1 = ftp_para_reuse->output_tiles[adjacent_task][l].h1;
      overlapped_region.up_region.h2 = all_region.h2;
      overlapped_region.up_region.w = overlapped_region.up_region.w2 - overlapped_region.up_region.w1 + 1;
      overlapped_region.up_region.h = overlapped_region.up_region.h2 - overlapped_region.up_region.h1 + 1;
      ftp_para_reuse->output_reuse_regions[adjacent_task][l] = overlapped_region;
#if DEBUG_FTP
      //printf("---(layer %3d), below---\n", l);
      printf("---(layer %3d), below---\n", l+ftp_para_reuse+1);
      print_tile_region(overlapped_region.up_region);
#endif
   }
   remaining_region.w = remaining_region.w2 - remaining_region.w1 + 1;
   remaining_region.h = remaining_region.h2 - remaining_region.h1 + 1;

  if (remaining_region.w <= 0) {
    // when the fused layer is deep, it is possible that adjacent tasks overlap
#if DEBUG_MULTI_FTP
    fprintf(stderr, "Warn: too deep fused layers causing fully overlap w...\n");
    exit(-1);
#endif
    remaining_region.w2 = remaining_region.w1; 
    remaining_region.w = 1;
  }
  if (remaining_region.h <= 0) {
#if DEBUG_MULTI_FTP
    fprintf(stderr, "Error: too deep fused layers causing fully overlap h...\n");
    exit(-1);
#endif
    remaining_region.h2 = remaining_region.h1; 
    remaining_region.h = 1;
  }

   return remaining_region;
}

void partition_and_estimate_reuse(network_parameters* net_para, ftp_parameters* ftp_para, ftp_parameters_reuse* ftp_para_reuse){
   uint32_t task;
   uint32_t data_size;
   uint32_t input_size = 0;
   uint32_t dw1, dw2;
   uint32_t dh1, dh2;
   uint32_t i, j;
   int32_t l;

   float comp_size;
   float comp_size_layer;
   float total_comp_size = 0;
   float original_total_comp_size = 0;

   uint32_t comm_size;
   uint32_t comm_size_layer;
   uint32_t total_comm_size = 0;
   uint32_t sp_layer_comm_size = 0;

   for(l = ftp_para->fused_layers-1; l >= 0; l--) {
     comm_size_layer = 0;
     comp_size_layer = 0;
     for(i = 0; i < ftp_para->partitions_h; i++){
       for(j = 0; j < ftp_para->partitions_w; j++){
         task = ftp_para->task_id[i][j];
         if (ftp_para_reuse->schedule[task] == 1) {
           int32_t n = net_para->n[l+ftp_para_reuse->from_layer];
           int32_t size = net_para->filter[l+ftp_para_reuse->from_layer];
           int32_t c1 = net_para->input_maps[l+ftp_para_reuse->from_layer].c;
           int32_t c2 = net_para->output_maps[l+ftp_para_reuse->from_layer].c;

           dw1 = ftp_para->output_tiles[task][l].w * ftp_para->output_tiles[task][l].h;
           dh1 = ftp_para_reuse->output_tiles[task][l].w * ftp_para_reuse->output_tiles[task][l].h;
           comm_size = sizeof(float)*(dw1-dh1)*c2;
           comp_size = ((2.0*n*size*size*c1) * (dw1-dh1)/1000000000.);
           comm_size_layer += comm_size;
           comp_size_layer += comp_size; 
         } 
       }
     }
     printf("Layer %u increased comm size: %u\n", l+ftp_para_reuse->from_layer, comm_size_layer);
     printf("Layer %u reduced comp size: %f (BFLOPs)\n", l+ftp_para_reuse->from_layer, comp_size_layer); 
     total_comm_size += comm_size_layer;
     total_comp_size += comp_size_layer;
   }
   // add comm size at sp layer
   sp_layer_comm_size += (ftp_para_reuse->partitions_w == 1 && ftp_para_reuse->partitions_h ==1) ? 0 : sizeof(float)*net_para->output_maps[ftp_para_reuse->from_layer+ftp_para_reuse->fused_layers-1].w 
     * net_para->output_maps[ftp_para_reuse->from_layer+ftp_para_reuse->fused_layers-1].h
     * net_para->output_maps[ftp_para_reuse->from_layer+ftp_para_reuse->fused_layers-1].c;
 
   printf("Layer [%u~%u), total_comm_size: %u (%fKB) vs. total_comp_size: %f (BFLOPs) comm_sp_layer: %u (%fKB)\n", ftp_para_reuse->from_layer, ftp_para_reuse->from_layer+ftp_para_reuse->fused_layers, total_comm_size, total_comm_size/1024., total_comp_size, sp_layer_comm_size, sp_layer_comm_size/1024.); 
}

void calculate_reuse_data_size(ftp_parameters_reuse* ftp_para_reuse, network_parameters* net_para, uint32_t task_id){

   uint32_t i = task_id/(ftp_para_reuse->partitions_w);
   uint32_t j = task_id%(ftp_para_reuse->partitions_w);
   int32_t adjacent_id[4];
   uint32_t position;
   uint32_t l;
   overlapped_tile_data regions_and_data;
   tile_region overlap_index;
   for(position = 0; position < 4; position++){
      adjacent_id[position] = -1;
   }

   /*position encoding
         2
         |
   3 <- self -> 1
         |
         0
   */

   /*get the up overlapped data from tile below*/
   if((i+1)<(ftp_para_reuse->partitions_h)) adjacent_id[0] = ftp_para_reuse->task_id[i+1][j];
   /*get the left overlapped data from tile on the right*/
   if((j+1)<(ftp_para_reuse->partitions_w)) adjacent_id[1] = ftp_para_reuse->task_id[i][j+1];
   /*get the bottom overlapped data from tile above*/
   if(i>0) adjacent_id[2] = ftp_para_reuse->task_id[i-1][j];
   /*get the right overlapped data from tile on the left*/
   if(j>0) adjacent_id[3] = ftp_para_reuse->task_id[i][j-1];

   ftp_para_reuse->adjacent_reuse_data_size[task_id]=0;
   ftp_para_reuse->self_reuse_data_size[task_id]=0;

   for(l = 0; l < ftp_para_reuse->fused_layers-1; l++){
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         uint32_t mirror_position = (position + 2)%4;
         regions_and_data = ftp_para_reuse->output_reuse_regions[adjacent_id[position]][l];
         overlap_index = get_region(&regions_and_data, mirror_position);
         if((overlap_index.w>0)&&(overlap_index.h>0))
            ftp_para_reuse->adjacent_reuse_data_size[task_id] += sizeof(float)*overlap_index.w*overlap_index.h*net_para->output_maps[l+ftp_para_reuse->from_layer].c;
      }
   }

   for(l = 0; l < ftp_para_reuse->fused_layers-1; l++){
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         regions_and_data = ftp_para_reuse->output_reuse_regions[task_id][l];
         overlap_index = get_region(&regions_and_data, position);
         if((overlap_index.w>0)&&(overlap_index.h>0))
            ftp_para_reuse->self_reuse_data_size[task_id] += sizeof(float)*overlap_index.w*overlap_index.h*net_para->output_maps[l+ftp_para_reuse->from_layer].c;
      }
   }
#if DEBUG_MULTI_FTP
   printf("adjacent_reuse_data_size for task %d: %u\n", task_id, ftp_para_reuse->adjacent_reuse_data_size[task_id]);
   printf("self_reuse_data_size for task %d: %u\n", task_id, ftp_para_reuse->self_reuse_data_size[task_id]);
#endif
}

/*This function must be called after perform_ftp()*/
ftp_parameters_reuse* preform_ftp_reuse(network_parameters* net_para, ftp_parameters* ftp_para){
   int32_t i, j, l;
   uint32_t task;
   uint32_t total_reuse_data_size = 0;

   ftp_parameters_reuse* ftp_para_reuse = (ftp_parameters_reuse*)malloc(sizeof(ftp_parameters_reuse));
   ftp_para_reuse->partitions = ftp_para->partitions;
   ftp_para_reuse->partitions_h = ftp_para->partitions_h;
   ftp_para_reuse->partitions_w = ftp_para->partitions_w;
   ftp_para_reuse->from_layer = ftp_para->from_layer;
   ftp_para_reuse->fused_layers = ftp_para->fused_layers;
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         ftp_para_reuse->task_id[i][j] = ftp_para->task_id[i][j];
      }
   }
   reuse_aware_schedule(ftp_para_reuse);

   /*Copy the grid output from normal ftp*/
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         task = ftp_para_reuse->task_id[i][j];
         l = ftp_para_reuse->fused_layers-1;
         ftp_para_reuse->output_tiles[task][l] = ftp_para->output_tiles[task][l];
      }
   }

   /*Calculate the tile regions with no reuse data dependency*/
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         task = ftp_para_reuse->task_id[i][j];
         for(l = ftp_para_reuse->fused_layers-1; l >= 0; l--){
            if(ftp_para_reuse->schedule[task] == 0){
               /*If there is no dependency, just copy from normal ftp parameters*/
               ftp_para_reuse->input_tiles[task][l] = ftp_para->input_tiles[task][l];
               ftp_para_reuse->output_tiles[task][l] = ftp_para->output_tiles[task][l];
            }
         }
      }
   }

   /*Calculate the tile regions with reuse data dependency*/
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         task = ftp_para_reuse->task_id[i][j];
         for(l = ftp_para_reuse->fused_layers-1; l >= 0; l--){
            ftp_para_reuse->output_reuse_regions[task][l].down_size = 0;
            ftp_para_reuse->output_reuse_regions[task][l].right_size = 0;
            ftp_para_reuse->output_reuse_regions[task][l].up_size = 0;
            ftp_para_reuse->output_reuse_regions[task][l].left_size = 0;
            ftp_para_reuse->output_reuse_regions[task][l].right_region.h = 0;
            ftp_para_reuse->output_reuse_regions[task][l].right_region.w = 0;
            ftp_para_reuse->output_reuse_regions[task][l].down_region.h = 0;
            ftp_para_reuse->output_reuse_regions[task][l].down_region.w = 0;
            ftp_para_reuse->output_reuse_regions[task][l].left_region.h = 0;
            ftp_para_reuse->output_reuse_regions[task][l].left_region.w = 0;
            ftp_para_reuse->output_reuse_regions[task][l].up_region.h = 0;
            ftp_para_reuse->output_reuse_regions[task][l].up_region.w = 0;
         }
      }
   }
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
#if DEBUG_FTP
         printf("----(%3d,%3d)----\n", i, j);
#endif
        task = ftp_para_reuse->task_id[i][j];
#if DEBUG_MULTI_FTP
        if(ftp_para_reuse->schedule[task] == 1)
          fprintf(stderr, "FTP_REUSE for task %d:\n", task);
#endif

         // offset: from_layer
         for(l = ftp_para_reuse->from_layer+ftp_para_reuse->fused_layers-1; l >= (int32_t) ftp_para_reuse->from_layer; l--){
            if(ftp_para_reuse->schedule[task] == 1){
#if DEBUG_MULTI_FTP
              fprintf(stderr, " layer %d:\t", l);
#endif
               ftp_para_reuse->input_tiles[ftp_para_reuse->task_id[i][j]][l-ftp_para_reuse->from_layer] = 
                       traversal(net_para, ftp_para_reuse->output_tiles[ftp_para_reuse->task_id[i][j]][l-ftp_para_reuse->from_layer], l);
               if(l>ftp_para_reuse->from_layer) 
                 ftp_para_reuse->output_tiles[ftp_para_reuse->task_id[i][j]][l-ftp_para_reuse->from_layer-1] = remove_and_record_overlapped_region_at_output(i, j, l-ftp_para_reuse->from_layer-1,  ftp_para_reuse, ftp_para_reuse->input_tiles[ftp_para_reuse->task_id[i][j]][l-ftp_para_reuse->from_layer]);
            }
#if DEBUG_FTP
            printf("---(layer %3d)---\n", l);
            print_tile_region(ftp_para_reuse->output_tiles[ftp_para->task_id[i][j]][l-ftp_para_reuse->from_layer]);
            print_tile_region(ftp_para_reuse->input_tiles[ftp_para->task_id[i][j]][l-ftp_para_reuse->from_layer]);
            printf("---(layer %3d)---\n", l);
#endif
         }
      }
   }

   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         task = ftp_para_reuse->task_id[i][j];
         calculate_reuse_data_size(ftp_para_reuse, net_para, task);/*Will be used in reuse_data serialization*/
         total_reuse_data_size += ftp_para_reuse->self_reuse_data_size[task];
      }
   }


#if DEBUG_MULTI_FTP
   printf("total reuse_data_size for current sp: %u\n", total_reuse_data_size);
#endif

   return ftp_para_reuse;
}

void set_coverage(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id){
   ftp_para_reuse->coverage[task_id]=1;
}

void set_missing(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id){
   ftp_para_reuse->missing[task_id]=1;
}

uint32_t get_missing(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id){
   return ftp_para_reuse->missing[task_id];
}

uint32_t get_coverage(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id){
   return ftp_para_reuse->coverage[task_id];
}

void clean_coverage(ftp_parameters_reuse* ftp_para_reuse){
   uint32_t task;
   uint32_t i, j;
   for(i = 0; i < ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < ftp_para_reuse->partitions_w; j++){
         task = ftp_para_reuse->task_id[i][j];
         ftp_para_reuse->coverage[task]=0;
         ftp_para_reuse->missing[task]=0;
      }
   }
}


bool is_reuse_ready(ftp_parameters_reuse* ftp_para_reuse, uint32_t task_id){
   uint32_t i = task_id/(ftp_para_reuse->partitions_w);
   uint32_t j = task_id%(ftp_para_reuse->partitions_w);
   uint32_t adj_task;
   bool ready = true;
   if(i + 1 < ftp_para_reuse->partitions_h){
      adj_task = ftp_para_reuse->task_id[i+1][j];
      if(ftp_para_reuse->coverage[adj_task] == 0) {
         ready = false;
         return ready;
      }	
   }
   if(j + 1 < ftp_para_reuse->partitions_w){
      adj_task = ftp_para_reuse->task_id[i][j+1];
      if(ftp_para_reuse->coverage[adj_task] == 0) {
         ready = false;
         return ready;
      }	
   }
   if(j > 0){
      adj_task = ftp_para_reuse->task_id[i][j-1];
      if(ftp_para_reuse->coverage[adj_task] == 0) {
         ready = false;
         return ready;
      }	
   }
   if(i > 0){
      adj_task = ftp_para_reuse->task_id[i-1][j];
      if(ftp_para_reuse->coverage[adj_task] == 0) {
         ready = false;
         return ready;
      }	
   }
   return ready;
}


/*position encoding
         2
         |
   3 <- self -> 1
         |
         0
*/

tile_region get_region(overlapped_tile_data * overlap, uint32_t pos){
   if(pos == 0) return overlap->down_region;
   if(pos == 1) return overlap->right_region;
   if(pos == 2) return overlap->up_region;
   if(pos == 3) return overlap->left_region;
   tile_region empty;
   return empty;
}

uint32_t get_size(overlapped_tile_data * overlap, uint32_t pos){
   if(pos == 0) return overlap->down_size;
   if(pos == 1) return overlap->right_size;
   if(pos == 2) return overlap->up_size;
   if(pos == 3) return overlap->left_size;
   return 0;
}

float* get_data(overlapped_tile_data * overlap, uint32_t pos){
   if(pos == 0) return overlap->down;
   if(pos == 1) return overlap->right;
   if(pos == 2) return overlap->up;
   if(pos == 3) return overlap->left;
   return NULL;
}

void set_region(overlapped_tile_data * overlap, uint32_t pos, tile_region tile){
   if(pos == 0) overlap->down_region = tile;
   if(pos == 1) overlap->right_region = tile;
   if(pos == 2) overlap->up_region = tile;
   if(pos == 3) overlap->left_region = tile;
}

void set_size(overlapped_tile_data * overlap, uint32_t pos, uint32_t size){
   if(pos == 0) overlap->down_size = size;
   if(pos == 1) overlap->right_size = size;
   if(pos == 2) overlap->up_size = size;
   if(pos == 3) overlap->left_size = size;
}

void set_data(overlapped_tile_data * overlap, uint32_t pos, float* data){
   if(pos == 0) overlap->down = data;
   if(pos == 1) overlap->right = data;
   if(pos == 2) overlap->up = data;
   if(pos == 3) overlap->left = data;
}
#endif /*DATA_REUSE*/
void print_tile_region(tile_region tile){
   printf("tile size is (%3d,%3d) \n", tile.w, tile.h);
   printf("(%3d,%3d)--------|\n", tile.w1, tile.h1);
   printf("|----------------|\n");
   printf("|----------------|\n");
   printf("|--------(%3d,%3d)\n", tile.w2, tile.h2);
}

