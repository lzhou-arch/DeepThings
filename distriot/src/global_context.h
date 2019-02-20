#ifndef GLOBAL_CONTEXT_H
#define GLOBAL_CONTEXT_H
#include "thread_safe_queue.h"
#include <string.h>
#define ADDR_LEN 64 
#define MAX_QUEUE_SIZE 256

typedef struct dev_ctxt {
   thread_safe_queue** results_pool;
   thread_safe_queue* ready_pool;
   uint32_t* results_counter;
   // temp results for split pointer is ready
   //uint32_t ready_sp;
   // temp results counter for split point
   uint32_t results_counter_sp;
   thread_safe_queue* registration_list;
   char** addr_list;
   uint32_t total_cli_num;

   thread_safe_queue* task_queue;
   thread_safe_queue* remote_task_queue;
   thread_safe_queue* result_queue; 
   // collect temp results
   thread_safe_queue* ready_queue; 
   uint32_t this_cli_id;

   uint32_t* batch_size_list;/*Number of tasks to merge (at each split point) */
   uint32_t batch_size;/*Number of tasks to merge*/
   void *model;/*pointers to execution model*/
   uint32_t total_frames;/*max number of input frames*/

   char gateway_local_addr[ADDR_LEN];
   char gateway_public_addr[ADDR_LEN];
   
   uint32_t is_gateway;
   uint32_t num_sp;
} device_ctxt;

device_ctxt* init_context(uint32_t cli_id, uint32_t cli_num, const char** edge_addr_list);
void set_batch_size(device_ctxt* ctxt, uint32_t size);
void set_batch_size_sp(device_ctxt* ctxt, uint32_t size, uint32_t i);
void set_gateway_local_addr(device_ctxt* ctxt, const char* addr);
void set_gateway_public_addr(device_ctxt* ctxt, const char* addr);
void set_total_frames(device_ctxt* ctxt, uint32_t frame_num);
#endif

