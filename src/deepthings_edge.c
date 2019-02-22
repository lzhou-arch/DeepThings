#include "deepthings_edge.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#if LOAD_AWARE 
#include "cpu.h"
#include "memory.h"
#endif
#if DEBUG_COMMU_SIZE
static double commu_size;
#endif

// alexnet
//#define FUSED_DEEPTH 8
//#define STOP_AT_LAYER 8 
// yolo
//#define FUSED_DEEPTH 25 
//#define STOP_AT_LAYER 25
// vgg-16
//#define FUSED_DEEPTH 18
//#define STOP_AT_LAYER 18
//
//

device_ctxt* deepthings_edge_estimate(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = init_client(edge_id, total_edge_number, addr_list);
   // don't load weights
   cnn_model* model = load_cnn_model(network, weights, 0, 0);

   // layer-wise
   layer_wise_overhead** layer_wise_overhead_list = layer_wise_estimate(model->net_para);

   // DP search for opt sol
   // store the opt layer result for each from_layer
   int32_t* dp_opt_fused_layers = (uint32_t*)malloc(sizeof(uint32_t)*model->net->n);
   float min_total_time = dp_buttom_up(0, model->net->n, model->net_para, layer_wise_overhead_list, dp_opt_fused_layers);

   printf("OPT: min total time: %f\n", min_total_time);
   printf("OPT fused points:\n"); 
   for (int32_t l = 0; l < model->net->n; l++) {
     int32_t opt_fused_layers = dp_opt_fused_layers[l];
     printf("[%d, %d) ", l, l+opt_fused_layers);
     print_dp_time(layer_wise_overhead_list, l, opt_fused_layers, 0);
     l += opt_fused_layers-1;
   }
   print_dp_time(layer_wise_overhead_list, 0, 0, 1);

   return ctxt;
}

device_ctxt* deepthings_edge_init(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = init_client(edge_id, total_edge_number, addr_list);
   // Load model weights from [0, last_fused_layers)
   cnn_model* model = load_cnn_model(network, weights, 0, from_layers[num_sp-1] + fused_layers[num_sp-1]);
   model->ftp_para_list = (ftp_parameters**)malloc(sizeof(ftp_parameters*)*num_sp);
   for (int i = 0; i < num_sp; i++) {
     model->ftp_para_list[i] = preform_ftp(N[i], M[i], from_layers[i], fused_layers[i], model->net_para);
     if (i > 0) {
       if (model->ftp_para_list[i-1]->from_layer+model->ftp_para_list[i-1]->fused_layers < model->ftp_para_list[i]->from_layer) {
         model->ftp_para_list[i-1]->gap_after = 1;
       }
     }
   }
   model->num_sp = num_sp;
   // set to fisrt sp
   model->ftp_para = model->ftp_para_list[0];
#if DATA_REUSE
   model->ftp_para_reuse_list = (ftp_parameters_reuse**)malloc(sizeof(ftp_parameters_reuse*)*num_sp);
   for (int i = 0; i < num_sp; i++) {
     model->ftp_para_reuse_list[i] = preform_ftp_reuse(model->net_para, model->ftp_para_list[i]);
   }
   model->ftp_para_reuse = model->ftp_para_reuse_list[0];
#endif
   ctxt->model = model;
   set_is_gateway(ctxt, 0);
   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   ctxt->batch_size_list = (uint32_t*)malloc(sizeof(uint32_t)*num_sp);
   for (int i = 0; i < num_sp; i++) {
     set_batch_size_sp(ctxt, N[i]*M[i], i);
   }
   // set to first sp
   set_batch_size(ctxt, N[0]*M[0]);
   set_num_sp(ctxt, num_sp);

   return ctxt;
}

#if DATA_REUSE
#if DATA_REUSE_LOCAL
static overlapped_tile_data* overlapped_data_pool_local[FUSED_POINTS_MAX][MAX_EDGE_NUM][PARTITIONS_MAX];

void save_local_reuse_data(device_ctxt* ctxt, blob* task_input_blob) {
  printf("saving_local_reuse_data ... ... \n");
  cnn_model* model = (cnn_model*)(ctxt->model);
  /*if task doesn't generate any reuse_data*/
  // in local sharing mode, every partition generates reuse_data for right and above partitions
  //if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

  blob* temp  = self_reuse_data_serialization(ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
  copy_blob_meta(temp, task_input_blob);

#if DEBUG_DEEP_EDGE
  printf("save self reuse data for task %d:%d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob), get_blob_sp_id(task_input_blob)); 
#endif

  int32_t cli_id;
  int32_t task_id;
  int32_t sp_id;

  cli_id = get_blob_cli_id(temp);
  task_id = get_blob_task_id(temp);
  sp_id = get_blob_sp_id(temp);
   
  //set_model_ftp_para(model, sp_id);

  printf("collecting_reuse_data ... ... \n");
  //if (model->cur_sp != sp_id) {
  //  fprintf(stderr, "Error: fuse point doesn't match (%d:%d)!\n", model->cur_sp, sp_id);
  //  exit(-1);
  //}

  if(overlapped_data_pool_local[sp_id][cli_id][task_id] != NULL)
     free_self_overlapped_tile_data(model,  overlapped_data_pool_local[sp_id][cli_id][task_id]);

  overlapped_data_pool_local[sp_id][cli_id][task_id] = self_reuse_data_deserialization(model, task_id, (float*)temp->data, get_blob_frame_seq(temp));

  printf("update_coverage ... ... \n");
  set_coverage(model->ftp_para_reuse, get_blob_task_id(temp));
  set_missing(model->ftp_para_reuse, get_blob_task_id(temp));
  free_blob(temp);

  return NULL;
}
#endif

void send_reuse_data(device_ctxt* ctxt, blob* task_input_blob){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;

   blob* temp  = self_reuse_data_serialization(ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   // TODO(lizhou): fix the static ip
   char local_addr[ADDR_LEN];
   strcpy(local_addr, "192.168.1.9");
   if(get_blob_sp_id(task_input_blob) == ctxt->num_sp-1) {
     fprintf(stderr, "Send reuse data to gateway ...\n");
     conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
   } else {
     fprintf(stderr, "Send reuse data to local ...\n");
     conn = connect_service(TCP, local_addr, WORK_STEAL_PORT);  // use the same port w/ steal
   }
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send self reuse data for task %d:%d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob), get_blob_sp_id(task_input_blob)); 
#endif
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}

void load_local_reuse_data(device_ctxt* ctxt, blob* task_input_blob, bool* reuse_data_is_required){
  printf("loading_reuse_data ... ... \n");
  cnn_model* model = (cnn_model*)(ctxt->model);
  /*if task doesn't require any reuse_data*/
  if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;/*Task without any dependency*/

  if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;/*Reuse data are all generated locally*/

  int32_t cli_id;
  int32_t task_id;
  int32_t sp_id;
  uint32_t frame_num;

  cli_id = get_blob_cli_id(task_input_blob);
  task_id = get_blob_task_id(task_input_blob);
  sp_id = get_blob_sp_id(task_input_blob);

  frame_num = get_blob_frame_seq(task_input_blob);

  set_model_ftp_para(model, sp_id);

  uint32_t position;
  int32_t* adjacent_id = get_adjacent_task_id_list(model, task_id);

  for(position = 0; position < 4; position++){
     if(adjacent_id[position]==-1) continue;
     if(reuse_data_is_required[position]){
        place_self_deserialized_data(model, adjacent_id[position], overlapped_data_pool_local[sp_id][cli_id][adjacent_id[position]]);
     }
  }
  free(adjacent_id);

/*
  // TODO(lizhou): remove serialization/deserialization steps
  blob* temp = adjacent_reuse_data_serialization(ctxt, task_id, frame_num, reuse_data_is_required);
  copy_blob_meta(temp, task_input_blob);
  overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
  place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);

  free_blob(temp);
*/
}

void request_reuse_data(device_ctxt* ctxt, blob* task_input_blob, bool* reuse_data_is_required){
   cnn_model* model = (cnn_model*)(ctxt->model);
   // update the ftp para reuse for requested reuse data
   int32_t cur_sp = get_blob_sp_id(task_input_blob);
   set_model_ftp_para_reuse(model, cur_sp);
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;/*Task without any dependency*/
   if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;/*Reuse data are all generated locally*/

   service_conn* conn;
   // TODO(lizhou): fix the static ip
   char local_addr[ADDR_LEN];
   strcpy(local_addr, "192.168.1.9");
   //conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
   
   fprintf(stderr, "check cur_sp %d, %u ...\n", cur_sp, ctxt->num_sp-1);

   if(cur_sp == ctxt->num_sp-1) {
     fprintf(stderr, "Request reuse data from gateway ...\n");
     conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
   } else {
     fprintf(stderr, "Request reuse data from local ...\n");
     conn = connect_service(TCP, local_addr, WORK_STEAL_PORT);  // use the same port w/ steal
   }
   send_request("request_reuse_data", 20, conn);
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif

   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);


   temp = recv_data(conn);
   copy_blob_meta(temp, task_input_blob);
   overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
   place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
   free_blob(temp);

   close_service_connection(conn);
}
#endif

// TODO(lizhou): make this as API
static inline void process_task(device_ctxt* ctxt, blob* temp, bool is_reuse){
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* result;
   set_model_input(model, (float*)temp->data);
   // update the ftp para for incoming tasks
   int32_t cur_sp = get_blob_sp_id(temp);
   set_model_ftp_para(model, cur_sp);
#if DATA_REUSE
   set_model_ftp_para_reuse(model, cur_sp);
#endif
   //fprintf(stderr, "Forwarding...\n");
   forward_partition(model, get_blob_task_id(temp), is_reuse);  
   result = new_blob_and_copy_data(0, 
              get_model_byte_size(model, model->ftp_para->from_layer+model->ftp_para->fused_layers-1), 
              (uint8_t*)(get_model_output(model, model->ftp_para->from_layer+model->ftp_para->fused_layers-1))
            );
#if DATA_REUSE
#if DATA_REUSE_LOCAL
   save_local_reuse_data(ctxt, temp);
#else
   send_reuse_data(ctxt, temp);
#endif
#endif
   copy_blob_meta(result, temp);
   enqueue(ctxt->result_queue, result); 
   free_blob(result);
}

// work in pull mode, send out task remotely
void send_task_data_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn* conn;
   blob* temp;
   int32_t task_counter = 0;
   while(1){
      // TODO(lizhou): can be sequential
      bool finish = true;
      for(int32_t i = 0; i < MAX_EDGE_NUM-1; i++) { 
        temp = try_dequeue(ctxt->remote_task_queues[i]);
        if(temp == NULL) continue;
        else {
          finish = false;

          int32_t dst_id = get_blob_ip_addr(temp);
          const char* cli_ip_addr = (const char*) ctxt->addr_list[dst_id];
    
          conn = connect_service(TCP, cli_ip_addr, WORK_STEAL_PORT);
          send_request("remote_exec", 20, conn);
    
#if DEBUG_FLAG
          task_counter ++;  
          printf("send_task data for task %d:%d, total number is %d\n", get_blob_cli_id(temp), get_blob_task_id(temp), task_counter); 
#endif
          send_data(temp, conn);
          free_blob(temp);
          close_service_connection(conn);
        }
      }
      if (finish) break;
   }
}

void partition_frame_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* model = (cnn_model*)(ctxt->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   uint32_t frame_num;
   bool* reuse_data_is_required;   
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      printf("Load image...\n");
      double time = sys_now_in_sec();
      /*Load image and partition, fill task queues*/
      load_image_as_model_input(model, frame_num);
      printf("Image loaded in: %lf seconds\n", sys_now_in_sec() - time);

      int processed_locally = 0;
      // process partitions between split points
      for (int i=0; i< (int)model->num_sp; i++) {
        printf("Process split point %d:%u ...\n", i, model->num_sp);
        
        time= sys_now_in_sec();
        if (i > 0) {
          if (processed_locally == 0) {
            // merge collected intermediate result and set model input 
            fprintf(stderr, "Check if temp results are ready at sp %u ...\n", model->cur_sp);
            blob* temp = local_dequeue_and_merge(ctxt);
#if DEBUG_FLAG
            int32_t cli_id = get_blob_cli_id(temp);
            int32_t frame_seq = get_blob_frame_seq(temp);
            int32_t sp_id = get_blob_frame_seq(temp);
            printf("Client %d, frame sequence number %d, split at %d, all partitions are merged locally\n", cli_id, frame_seq, sp_id);
#endif
            float* fused_output = (float*)(temp->data);
            // keep output for the last fused layer
            //set_model_output(model, model->ftp_para->from_layer+model->ftp_para->fused_layers-1, fused_output);
            set_model_input(model, fused_output);
          } else {
            // input is from local layer
            set_model_input(model, model->net->layers[model->ftp_para->from_layer+model->ftp_para->fused_layers-1].output);
            processed_locally = 0;
          }
          if (model->ftp_para->from_layer+model->ftp_para->fused_layers < model->ftp_para_list[i]->from_layer && model->ftp_para->gap_after == 1) {
            fprintf(stderr, "Warning: Non fused layers in between [%d, %d), keep calm and carry on...\n", model->ftp_para->from_layer+model->ftp_para->fused_layers,
                model->ftp_para_list[i]->from_layer);
            double time = sys_now_in_sec();
            forward_from_upto(model, model->ftp_para->from_layer+model->ftp_para->fused_layers, model->ftp_para_list[i]->from_layer);
            printf("Process non-fused [%d, %d) layers in: %f\n",
                model->ftp_para->from_layer+model->ftp_para->fused_layers,
                model->ftp_para_list[i]->from_layer,
                sys_now_in_sec() - time);
            i--;  // rollback
            model->ftp_para->gap_after = 0;  // set gap is done
            processed_locally = 1;
            continue;
          }
        }

        // set model ftp para 
        model->cur_sp = i;
        model->ftp_para = model->ftp_para_list[model->cur_sp];
#if DATA_REUSE
        model->ftp_para_reuse = model->ftp_para_reuse_list[model->cur_sp];
#endif
        ctxt->batch_size = ctxt->batch_size_list[model->cur_sp];

        fprintf(stderr, "Temp results are ready at sp %u ...\n", model->cur_sp);
        partition_and_enqueue(ctxt, frame_num);

        // first split point, read input from image and register workload.
        
#if POLL_MODE
        if (i==0) register_client(ctxt);
#endif
        printf("Input before split point %d partitioned in: %lf seconds\n", i, sys_now_in_sec() - time);

        // send out remote task
        sys_thread_t t3 = sys_thread_new("send_task_data_thread", send_task_data_thread, ctxt, 0, 0);

        time = sys_now_in_sec();
        int num_task = 0;
        /*Dequeue and process task*/
        while(1){
           temp = try_dequeue(ctxt->task_queue);
           if(temp == NULL) break;

           // TODO(lizhou): check if remote or local and send in another thread
           //int32_t dst_id = get_blob_ip_addr(temp);
           //if (dst_id > 0) {
           //  send_task_data(ctxt, temp, dst_id);
           //  continue;
           //}

           bool data_ready = false;
#if DEBUG_DEEP_EDGE
           printf("====================Processing task id is %d, data source is %d, frame_seq is %d, sp_id is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp), get_blob_sp_id(temp));
#endif/*DEBUG_DEEP_EDGE*/
#if DATA_REUSE
           data_ready = is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp));
           if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && data_ready) {
              blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                         (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                         (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
              copy_blob_meta(shrinked_temp, temp);
              free_blob(temp);
              temp = shrinked_temp;

              reuse_data_is_required = check_missing_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
#if DEBUG_DEEP_EDGE
              printf("Request data from gateway, is there anything missing locally? ...\n");
              print_reuse_data_is_required(reuse_data_is_required);
#endif/*DEBUG_DEEP_EDGE*/
#if DATA_REUSE_LOCAL
              // don't need to request if process locally
              load_local_reuse_data(ctxt, temp, reuse_data_is_required);
#else
              request_reuse_data(ctxt, temp, reuse_data_is_required);
#endif
              free(reuse_data_is_required);
           }
#if DEBUG_DEEP_EDGE
           if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && (!data_ready))
              printf("The reuse data is not ready yet!\n");
#endif/*DEBUG_DEEP_EDGE*/

#endif/*DATA_REUSE*/
#if DEBUG_TIMING
           double time_tmp = sys_now_in_sec();
#endif
           process_task(ctxt, temp, data_ready);
#if DEBUG_TIMING
           time_tmp = sys_now_in_sec() - time_tmp;
           printf("Process task in: %f\n", time_tmp);
           num_task++;
#endif
           free_blob(temp);
#if DEBUG_COMMU_SIZE
           printf("======Communication size at edge is: %f======\n", ((double)commu_size)/(1024.0*1024.0*FRAME_NUM));
#endif
        }

        // sync with remote tasks
        sys_thread_join(t3);

#if DEBUG_TIMING
        printf("Early cnn before split point %d processed in: %lf (avg: %lf)\n", i, sys_now_in_sec() - time, (sys_now_in_sec() - time) / num_task);
#endif
      }
#if POLL_MODE
      /*Unregister and prepare for next image*/
      cancel_client(ctxt);
#endif
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}

// collect temp result to source device 
void* edge_result_source(void* srv_conn, void* arg){
   printf("result_source ... ... \n");
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn *conn = (service_conn *)srv_conn;
   int32_t cli_id;
   int32_t frame_seq;
   int32_t sp_id;
#if DEBUG_FLAG
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, ctxt);
#if DEBUG_TIMING
   double total_time;
   uint32_t total_frames;
   double now;
   uint32_t i;
#endif
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   frame_seq = get_blob_frame_seq(temp);
   sp_id = get_blob_sp_id(temp);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
#if DEBUG_FLAG
   printf("Result from %d: %s is for client %d of sp %d, total number recved is %d\n", processing_cli_id, ip_addr, cli_id, sp_id, ctxt->results_counter_sp+1);
#endif
   enqueue(ctxt->ready_queue, temp);
   free_blob(temp);
   ctxt->results_counter_sp++;
   if(ctxt->results_counter_sp == ctxt->batch_size){
      temp = new_empty_blob(cli_id);
#if DEBUG_FLAG
      printf("Results for client %d are all collected in edge_result_source, update ready_pool\n", cli_id);
#endif
#if DEBUG_TIMING
      printf("Client %d, frame sequence number %d, all partitions are merged in local_merge_result_thread\n", cli_id, frame_seq);
      now = sys_now_in_sec();
      /*Total latency*/
/*
      acc_time[cli_id] = now - start_time;
      acc_frames[cli_id] = frame_seq + 1;
      total_time = 0;
      total_frames = 0;
      for(i = 0; i < ctxt->total_cli_num; i ++){
         if(acc_frames[i] > 0)
             printf("Avg latency for Client %d at sp %d is: %f\n", i, sp_id, acc_time[i]/acc_frames[i]);
         total_time = total_time + acc_time[i];
         total_frames = total_frames + acc_frames[i];
      }
      printf("Avg latency for all clients %f\n", total_time/total_frames);
      */
#endif
#if DEBUG_COMMU_SIZE
      printf("Communication size at gateway is: %f\n", ((double)commu_size)/(1024.0*1024.0*FRAME_NUM));
#endif
      fprintf(stderr, "Temp results are ready with %d batches\n", ctxt->batch_size);
      enqueue(ctxt->ready_pool, temp);
      //ctxt->ready_sp = 1;
      free_blob(temp);
      ctxt->results_counter_sp = 0;
   }

   return NULL;
}

void edge_collect_result_thread(void *arg) {
  const char* request_types[] = {"result_source"};
  void* (*handlers[])(void*, void*) = {edge_result_source};
  int result_service = service_init(RESULT_COLLECT_PORT + 10, TCP); // port for local
  start_service(result_service, TCP, request_types, 1, handlers, arg);
  close_service(result_service);
}

void steal_partition_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   /*Check gateway for possible stealing victims*/
#ifdef NNPACK
   cnn_model* model = (cnn_model*)(ctxt->model);
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   service_conn* conn;
   blob* temp;
   double time_tmp;
   double total_exec_time = 0;
   double total_fetch_time = 0;
   int num_task = 0;
   while(1){
#if LOAD_AWARE
      // check stats e.g. cpu/memory
      struct MemoryStatus status;
      mem_status(&status);
      float mem_load = status.used_mem / status.total_mem;
      float cpu_load = cpu_percentage(CPU_USAGE_DELAY);
      printf("cpu load: %f, mem load: %f\n", cpu_load, mem_load);
      if (cpu_load > MAX_CPU_LOAD || mem_load > MAX_MEM_LOAD) {
        printf("cpu/mem load is too high\n");
        sys_sleep(100);
        continue;
      }
#endif
      conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
      send_request("steal_gateway", 20, conn);
#if LOAD_AWARE
      // send stats
      temp = new_empty_blob(110);
      send_data(temp, conn);
#endif
      temp = recv_data(conn);
      close_service_connection(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      
      time_tmp = sys_now_in_sec();
      // connect to source and steal workload
      const char *cli_ip_addr = (const char *)temp->data;
      conn = connect_service(TCP, (const char *)temp->data, WORK_STEAL_PORT);
      send_request("steal_client", 20, conn);
      free_blob(temp);
      temp = recv_data(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      bool data_ready = true;
#if DATA_REUSE
      blob* reuse_info_blob = recv_data(conn);
      bool* reuse_data_is_required = (bool*) reuse_info_blob->data;
      request_reuse_data(ctxt, temp, reuse_data_is_required);
      if(!need_reuse_data_from_gateway(reuse_data_is_required)) data_ready = false; 
#if DEBUG_DEEP_EDGE
      printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
      printf("Request data from gateway, is the reuse data ready? ...\n");
      print_reuse_data_is_required(reuse_data_is_required);
#endif

      free_blob(reuse_info_blob);
#endif
      close_service_connection(conn);
      num_task++;
#if DEBUG_TIMING
      time_tmp = sys_now_in_sec() - time_tmp;
      total_fetch_time +=  time_tmp;
      printf("Fetch task in: %lf (total: %lf/avg: %lf)\n", time_tmp, total_fetch_time, total_fetch_time/ num_task);
#endif
      time_tmp = sys_now_in_sec();
      process_task(ctxt, temp, data_ready);
#if DEBUG_TIMING
      time_tmp = sys_now_in_sec() - time_tmp;
      total_exec_time +=  time_tmp;
      printf("Process task in: %lf (total: %lf/avg: %lf)\n", time_tmp, total_exec_time, total_exec_time / num_task);
#endif
      free_blob(temp);
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}

static double total_remote_exec_time, remote_num_task;

void remote_exec(void* srv_conn, void* arg) {
   printf("remote_exec... ... \n");
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn *conn = (service_conn *)srv_conn;
#ifdef NNPACK
   cnn_model* model = (cnn_model*)(ctxt->model);
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

   int32_t cli_id;
   int32_t frame_seq;
   int32_t sp_id;
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   frame_seq = get_blob_frame_seq(temp);
   sp_id = get_blob_sp_id(temp);

   set_model_ftp_para(model, sp_id);

   double time_tmp = sys_now_in_sec();
   bool data_ready = false;
#if DATA_REUSE
   data_ready = is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp));
   if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && data_ready) {
     blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp),
         (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
         (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
     copy_blob_meta(shrinked_temp, temp);
     free_blob(temp);
     temp = shrinked_temp;

     //bool* reuse_data_is_required = (bool*)malloc(sizeof(bool)*4);
     bool* reuse_data_is_required = check_missing_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
#if DEBUG_DEEP_EDGE
     printf("Request data from gateway, is there anything missing locally? ...\n");
     print_reuse_data_is_required(reuse_data_is_required);
#endif/*DEBUG_DEEP_EDGE*/
#if DATA_REUSE_LOCAL
     // don't need to request if process locally
     load_local_reuse_data(ctxt, temp, reuse_data_is_required);
#else
     request_reuse_data(ctxt, temp, reuse_data_is_required);
#endif
     free(reuse_data_is_required);
   }
#if DEBUG_DEEP_EDGE
   if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && (!data_ready))
     printf("The reuse data is not ready yet!\n");
#endif/*DEBUG_DEEP_EDGE*/
#endif/*DATA_REUSE*/

//#if DATA_REUSE
//   blob* reuse_info_blob = recv_data(conn);
//   bool* reuse_data_is_required = (bool*) reuse_info_blob->data;
//   request_reuse_data(ctxt, temp, reuse_data_is_required);
//   if(!need_reuse_data_from_gateway(reuse_data_is_required)) data_ready = false; 
//#if DEBUG_DEEP_EDGE
//   printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
//   printf("Request data from gateway, is the reuse data ready? ...\n");
//   print_reuse_data_is_required(reuse_data_is_required);
//#endif
//
//   free_blob(reuse_info_blob);
//#endif
   //close_service_connection(conn);
   remote_num_task++;
//#if DEBUG_TIMING
//   time_tmp = sys_now_in_sec() - time_tmp;
//   total_fetch_time +=  time_tmp;
//   printf("Fetch task in: %lf (total: %lf/avg: %lf)\n", time_tmp, total_fetch_time, total_fetch_time/ num_task);
//#endif
   //time_tmp = sys_now_in_sec();
   process_task(ctxt, temp, data_ready);
#if DEBUG_TIMING
   time_tmp = sys_now_in_sec() - time_tmp;
   total_remote_exec_time +=  time_tmp;
   printf("Process task in: %lf (total: %lf/avg: %lf)\n", time_tmp, total_remote_exec_time, total_remote_exec_time / remote_num_task);
#endif
   free_blob(temp);

#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
   //return NULL;
}


/*defined in gateway.h from darkiot
void send_result_thread;
*/


/*Function handling steal reqeust*/
#if DATA_REUSE
void* steal_client_reuse_aware(void* srv_conn, void* arg){
   printf("steal_client_reuse_aware ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = try_dequeue(ctxt->task_queue);
   if(temp == NULL){
      char data[20]="empty";
      temp = new_blob_and_copy_data(-1, 20, (uint8_t*)data);
      send_data(temp, conn);
      free_blob(temp);
      return NULL;
   }
#if DEBUG_DEEP_EDGE
   printf("Stolen local task is %d\n", temp->id);
#endif

   uint32_t task_id = get_blob_task_id(temp);
   // set ftp para reuse when requesting task
   uint32_t sp_id = get_blob_sp_id(temp);
   set_model_ftp_para_reuse(edge_model, sp_id);
   bool* reuse_data_is_required = (bool*)malloc(sizeof(bool)*4);
   uint32_t position;
   for(position = 0; position < 4; position++){
      reuse_data_is_required[position] = false;
   }

   if(edge_model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(edge_model->ftp_para_reuse, get_blob_task_id(temp))) {
      uint32_t position;
      int32_t* adjacent_id = get_adjacent_task_id_list(edge_model, task_id);
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         reuse_data_is_required[position] = true;
      }
      free(adjacent_id);
      blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (edge_model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(edge_model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
      copy_blob_meta(shrinked_temp, temp);
      free_blob(temp);
      temp = shrinked_temp;
   }
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   /*Send bool variables for different positions*/
   temp = new_blob_and_copy_data(task_id, 
                       sizeof(bool)*4,
                       (uint8_t*)(reuse_data_is_required));
   free(reuse_data_is_required);
   send_data(temp, conn);
   free_blob(temp);

   return NULL;
}

void* update_coverage(void* srv_conn, void* arg){
   printf("update_coverage ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = recv_data(conn);
#if DEBUG_DEEP_EDGE
   printf("set coverage for task %d\n", get_blob_task_id(temp));
#endif
   set_coverage(edge_model->ftp_para_reuse, get_blob_task_id(temp));
   set_missing(edge_model->ftp_para_reuse, get_blob_task_id(temp));
   free_blob(temp);
   return NULL;
}

void notify_coverage_by_ip(device_ctxt* ctxt, blob* task_input_blob, uint32_t cli_id, char* cli_addr){
   printf("notify coverage by ip...\n");
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(cli_id, 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   service_conn* conn;
   conn = connect_service(TCP, cli_addr, WORK_STEAL_PORT);
   send_request("update_coverage", 20, conn);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}

//static overlapped_tile_data* overlapped_data_pool_local[FUSED_POINTS_MAX][MAX_EDGE_NUM][PARTITIONS_MAX];

void* recv_reuse_data_from_edge_local(void* srv_conn, void* arg){
   printf("collecting_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   cnn_model* gateway_model = (cnn_model*)(((device_ctxt*)(arg))->model);

   int32_t cli_id;
   int32_t task_id;
   int32_t sp_id;

   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   // only works for gateway
   processing_cli_id = get_client_id(ip_addr, (device_ctxt*)arg);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");

   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   sp_id = get_blob_sp_id(temp);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d at sp %d is collected from %d: %s, size is %d\n", cli_id, task_id, sp_id, processing_cli_id, ip_addr, temp->size);
#endif

   // set the correct sp ftp_para_reuse
   //gateway_model->ftp_para_reuse = gateway_model->ftp_para_reuse_list[sp_id];
   //if (gateway_model->cur_sp != sp_id) {
   //  fprintf(stderr, "Error: fuse point doesn't match!\n");
   //  exit(-1);
   //}

   if(overlapped_data_pool_local[sp_id][cli_id][task_id] != NULL)
      free_self_overlapped_tile_data(gateway_model,  overlapped_data_pool_local[sp_id][cli_id][task_id]);

   overlapped_data_pool_local[sp_id][cli_id][task_id] = self_reuse_data_deserialization(gateway_model, task_id, (float*)temp->data, get_blob_frame_seq(temp));

   // TODO(lizhou): record this cli ip addr
   //if(processing_cli_id != cli_id) notify_coverage((device_ctxt*)arg, temp, cli_id);
   if(processing_cli_id != cli_id) notify_coverage_by_ip((device_ctxt*)arg, temp, cli_id, "192.168.0.100");
   //if(processing_cli_id != cli_id) notify_coverage_by_ip((device_ctxt*)arg, temp, cli_id, "192.168.1.9");
   free_blob(temp);

#if DEBUG_DEEP_GATEWAY
   printf("Reuse data is received!\n");
#endif


   return NULL;
}

void* send_reuse_data_to_edge_local(void* srv_conn, void* arg){
   printf("handing_out_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* gateway_model = (cnn_model*)(ctxt->model);

   int32_t cli_id;
   int32_t task_id;
   int32_t sp_id;
   uint32_t frame_num;
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   sp_id = get_blob_sp_id(temp);

   frame_num = get_blob_frame_seq(temp);
   free_blob(temp);

#if DEBUG_DEEP_GATEWAY
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, ctxt);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif

   blob* reuse_info_blob = recv_data(conn);
   bool* reuse_data_is_required = (bool*)(reuse_info_blob->data);

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is required by %d: %s is \n", cli_id, task_id, processing_cli_id, ip_addr);
   print_reuse_data_is_required(reuse_data_is_required);
#endif
   uint32_t position;
   int32_t* adjacent_id = get_adjacent_task_id_list(gateway_model, task_id);

   for(position = 0; position < 4; position++){
      if(adjacent_id[position]==-1) continue;
      if(reuse_data_is_required[position]){
#if DEBUG_DEEP_GATEWAY
         printf("place_self_deserialized_data for client %d, task %d, the adjacent task is %d\n", cli_id, task_id, adjacent_id[position]);
#endif
         place_self_deserialized_data(gateway_model, adjacent_id[position], overlapped_data_pool_local[sp_id][cli_id][adjacent_id[position]]);
      }
   }
   free(adjacent_id);
   temp = adjacent_reuse_data_serialization(ctxt, task_id, frame_num, reuse_data_is_required);
   free_blob(reuse_info_blob);
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   return NULL;
}
#endif

void recv_partition_and_perform_inference_thread(void *arg){
   const char* request_types[]={"remote_exec"};
   void* (*handlers[])(void*, void*) = {remote_exec};

   int wst_service = service_init(WORK_STEAL_PORT, TCP);
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
   close_service(wst_service);
}

void deepthings_serve_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"steal_client", "update_coverage", "reuse_data", "request_reuse_data"};
   void* (*handlers[])(void*, void*) = {steal_client_reuse_aware, update_coverage, recv_reuse_data_from_edge_local, send_reuse_data_to_edge_local};
#else
   const char* request_types[]={"steal_client"};
   void* (*handlers[])(void*, void*) = {steal_client};
#endif
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   //start_service(wst_service, TCP, request_types, 2, handlers, arg);
   start_service(wst_service, TCP, request_types, 4, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
#endif
   close_service(wst_service);
}

void send_task_data(device_ctxt* ctxt, blob* temp, int32_t dst_id){
  service_conn* conn;
  const char* cli_ip_addr = (const char*) ctxt->addr_list[dst_id];

  fprintf(stderr, "Send to %c...\n", cli_ip_addr);
  conn = connect_service(TCP, cli_ip_addr, WORK_STEAL_PORT);
  send_request("remote_exec", 20, conn);

#if DEBUG_FLAG
  printf("send_task data for task %d:%d\n", get_blob_cli_id(temp), get_blob_task_id(temp)); 
#endif
  send_data(temp, conn);
  free_blob(temp);
  close_service_connection(conn);
}

void deepthings_stealer_edge(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t total_edge_number, const char** addr_list){

   device_ctxt* ctxt = deepthings_edge_init(num_sp, N, M, from_layers, fused_layers, network, weights, edge_id, total_edge_number, addr_list);
   //exec_barrier(START_CTRL, TCP, ctxt);
   exec_barrier_edge(START_CTRL, TCP, ctxt);

#if POLL_MODE
   sys_thread_t t1 = sys_thread_new("steal_partition_and_perform_inference_thread", steal_partition_and_perform_inference_thread, ctxt, 0, 0);
#else  // Push mode
   sys_thread_t t1 = sys_thread_new("recv_partition_and_perform_inference_thread", recv_partition_and_perform_inference_thread, ctxt, 0, 0);
#endif
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
}

void deepthings_victim_edge(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t total_edge_number, const char** addr_list){

   device_ctxt* ctxt = deepthings_edge_init(num_sp, N, M, from_layers, fused_layers, network, weights, edge_id, total_edge_number, addr_list);
   //exec_barrier(START_CTRL, TCP, ctxt);
   exec_barrier_edge(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);
#if POLL_MODE
   sys_thread_t t3 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, ctxt, 0, 0);
#else  // Push mode
   //sys_thread_t t3 = sys_thread_new("send_task_data_thread", send_task_data_thread, ctxt, 0, 0);
#endif

   sys_thread_t t4 = sys_thread_new("edge_collect_result_thread", edge_collect_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   //sys_thread_join(t3);
   sys_thread_join(t4);
}

void deepthings_estimate(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = deepthings_edge_estimate(num_sp, N, M, from_layers, fused_layers, network, weights, edge_id, total_edge_number, addr_list);
}
