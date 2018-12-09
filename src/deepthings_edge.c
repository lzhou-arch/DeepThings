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

device_ctxt* deepthings_edge_init(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id){
   device_ctxt* ctxt = init_client(edge_id);
   // Load model weights from [0, last_fused_layers)
   cnn_model* model = load_cnn_model(network, weights, 0, from_layers[num_sp-1] + fused_layers[num_sp-1]);
   model->ftp_para_list = (ftp_parameters**)malloc(sizeof(ftp_parameters*)*num_sp);
   for (int i = 0; i < num_sp; i++) {
     model->ftp_para_list[i] = preform_ftp(N[i], M[i], from_layers[i], fused_layers[i], model->net_para);
   }
   model->num_sp = num_sp;
   // set to fisrt sp
   model->ftp_para = model->ftp_para_list[0];
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
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
void send_reuse_data(device_ctxt* ctxt, blob* task_input_blob){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;

   blob* temp  = self_reuse_data_serialization(ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send self reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}

void request_reuse_data(device_ctxt* ctxt, blob* task_input_blob, bool* reuse_data_is_required){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;/*Task without any dependency*/
   if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;/*Reuse data are all generated locally*/

   service_conn* conn;
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT + 10);
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
   forward_partition(model, get_blob_task_id(temp), is_reuse);  
   result = new_blob_and_copy_data(0, 
              get_model_byte_size(model, model->ftp_para->from_layer+model->ftp_para->fused_layers-1), 
              (uint8_t*)(get_model_output(model, model->ftp_para->from_layer+model->ftp_para->fused_layers-1))
            );
#if DATA_REUSE
   send_reuse_data(ctxt, temp);
#endif
   copy_blob_meta(result, temp);
   enqueue(ctxt->result_queue, result); 
   free_blob(result);
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

      // process partitions between split points
      for (int i=0; i< (int)model->num_sp; i++) {
        printf("Process split point %d:%u ...\n", i, model->num_sp);
        
        time= sys_now_in_sec();
        if (i > 0) {
          fprintf(stderr, "Check if temp results are ready at sp %u ...\n", model->cur_sp);
          // merge collected intermediate result and set model input 
          blob* temp = local_dequeue_and_merge(ctxt);
#if DEBUG_FLAG
          int32_t cli_id = get_blob_cli_id(temp);
          int32_t frame_seq = get_blob_frame_seq(temp);
          int32_t sp_id = get_blob_frame_seq(temp);
          printf("Client %d, frame sequence number %d, split at %d, all partitions are merged locally\n", cli_id, frame_seq, sp_id);
#endif
          float* fused_output = (float*)(temp->data);
          set_model_input(model, fused_output);
          //double time = sys_now_in_sec();
          //printf("Test forward all");
          //forward_all(model, model->ftp_para_list[0]->fused_layers);
          //printf(" in: %f\n", sys_now_in_sec() - time);
          //exit(-1);
        }

        // set model ftp para 
        model->cur_sp = i;
        model->ftp_para = model->ftp_para_list[model->cur_sp];
        ctxt->batch_size = ctxt->batch_size_list[model->cur_sp];

        fprintf(stderr, "Temp results are ready at sp %u ...\n", model->cur_sp);
        partition_and_enqueue(ctxt, frame_num);

        // first split point, read input from image and register workload.
        if (i==0) register_client(ctxt);
        printf("Input before split point %d partitioned in: %lf seconds\n", i, sys_now_in_sec() - time);

        time = sys_now_in_sec();
        int num_task = 0;
        /*Dequeue and process task*/
        while(1){
           temp = try_dequeue(ctxt->task_queue);
           if(temp == NULL) break;
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
              request_reuse_data(ctxt, temp, reuse_data_is_required);
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
#if DEBUG_TIMING
        printf("Early cnn before split point %d processed in: %lf (avg: %lf)\n", i, sys_now_in_sec() - time, (sys_now_in_sec() - time) / num_task);
#endif
      }
      /*Unregister and prepare for next image*/
      cancel_client(ctxt);
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
#endif

// TODO(lizhou)
//void recv_partition_and_perform_inference_thread(void *arg){
//   const char* request_types[]={"remote_exec"};
//   void* (*handlers[])(void*, void*) = {remote_exec};
//
//   int wst_service = service_init(WORK_STEAL_PORT, TCP);
//   start_service(wst_service, TCP, request_types, 1, handlers, arg);
//   close_service(wst_service);
//}

void deepthings_serve_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"steal_client", "update_coverage"};
   void* (*handlers[])(void*, void*) = {steal_client_reuse_aware, update_coverage};
#else
   const char* request_types[]={"steal_client"};
   void* (*handlers[])(void*, void*) = {steal_client};
#endif
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 2, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
#endif
   close_service(wst_service);
}

void deepthings_stealer_edge(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id){

   device_ctxt* ctxt = deepthings_edge_init(num_sp, N, M, from_layers, fused_layers, network, weights, edge_id);
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

void deepthings_victim_edge(uint32_t num_sp, uint32_t* N, uint32_t* M, uint32_t* from_layers, uint32_t* fused_layers, char* network, char* weights, uint32_t edge_id){

   device_ctxt* ctxt = deepthings_edge_init(num_sp, N, M, from_layers, fused_layers, network, weights, edge_id);
   //exec_barrier(START_CTRL, TCP, ctxt);
   exec_barrier_edge(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);
#if POLL_MODE
   sys_thread_t t3 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, ctxt, 0, 0);
#else  // Push mode
   sys_thread_t t3 = sys_thread_new("deepthings_task_sharing_thread", deepthings_task_sharing_thread, ctxt, 0, 0);
#endif

   sys_thread_t t4 = sys_thread_new("edge_collect_result_thread", edge_collect_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
   sys_thread_join(t4);
}

