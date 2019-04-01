#include "darkiot.h"
#include "schedule.h"
#include "frame_partitioner.h"

void partition_and_enqueue(device_ctxt* ctxt, uint32_t frame_num){
   cnn_model* model = (cnn_model*)(ctxt->model);
   uint32_t task;
   network_parameters* net_para = model->net_para;
   float* data;
   uint32_t data_size;
   blob* temp;
   uint32_t dw1, dw2;
   uint32_t dh1, dh2;
   uint32_t i, j;
   uint32_t total_tasks = model->ftp_para->partitions_h*model->ftp_para->partitions_w;
   uint32_t num_devices = model->ftp_para->num_devices;

   for(i = 0; i < model->ftp_para->partitions_h; i++){
      for(j = 0; j < model->ftp_para->partitions_w; j++){
         task = model->ftp_para->task_id[i][j];
         dw1 = model->ftp_para->input_tiles[task][0].w1;
         dw2 = model->ftp_para->input_tiles[task][0].w2;
         dh1 = model->ftp_para->input_tiles[task][0].h1;
         dh2 = model->ftp_para->input_tiles[task][0].h2;
         data = crop_feature_maps(get_model_input(model), 
                                  net_para->input_maps[model->ftp_para->from_layer].w, 
                                  net_para->input_maps[model->ftp_para->from_layer].h,
                                  net_para->input_maps[model->ftp_para->from_layer].c, 
                                  dw1, dw2, dh1, dh2);
         data_size = sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->input_maps[model->ftp_para->from_layer].c;
         temp = new_blob_and_copy_data((int32_t)task, data_size, (uint8_t*)data);
         free(data);
#if POLL_MODE
         annotate_blob(temp, get_this_client_id(ctxt), frame_num, task, model->cur_sp);
#else
         // TODO(lizhou): add function that assign tasks to each device
         int32_t dst_id = get_task_dst_id(task, total_tasks, num_devices);
         //fprintf(stderr, "Assign task %u to dev %u..\n", task, dst_id);
         if (dst_id < 0) fprintf(stderr, "Error: check the assigned device id.\n");
         annotate_blob_push(temp, get_this_client_id(ctxt), frame_num, task, model->cur_sp, dst_id);
#endif

#if POLL_MODE
         enqueue(ctxt->task_queue, temp);
#else
         // push into remote queue
         if (dst_id > 0)
           enqueue(ctxt->remote_task_queues[dst_id-1], temp);
         else  // push into local queue
           enqueue(ctxt->task_queue, temp);
#endif
         free_blob(temp);
         printf("Task %u, size: %u\n", task, data_size); 
      }
   }

#if POLL_MODE
#if DATA_REUSE

   for(i = 0; i < model->ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < model->ftp_para_reuse->partitions_w; j++){
         task = model->ftp_para_reuse->task_id[i][j];
         if(model->ftp_para_reuse->schedule[task] == 1){
            // moved to later part of queue
            remove_by_id(ctxt->task_queue, task);
            /*Enqueue original size for rollback execution if adjacent partition is not ready... ...*/
            dw1 = model->ftp_para->input_tiles[task][0].w1;
            dw2 = model->ftp_para->input_tiles[task][0].w2;
            dh1 = model->ftp_para->input_tiles[task][0].h1;
            dh2 = model->ftp_para->input_tiles[task][0].h2;
            data = crop_feature_maps(get_model_input(model), 
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].w, 
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].h,
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].c, 
                                  dw1, dw2, dh1, dh2);
            data_size = sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->input_maps[model->ftp_para_reuse->from_layer].c;
            temp = new_blob_and_copy_data((int32_t)task, data_size, (uint8_t*)data);
            free(data);
            annotate_blob(temp, get_this_client_id(ctxt), frame_num, task, model->cur_sp);
            enqueue(ctxt->task_queue, temp);
            free_blob(temp);
        }
      }
   }
#endif  // DATA_REUSE
#endif  // POLL_MODE

#if DATA_REUSE
   clean_coverage(model->ftp_para_reuse);
   for(i = 0; i < model->ftp_para_reuse->partitions_h; i++){
      for(j = 0; j < model->ftp_para_reuse->partitions_w; j++){
         task = model->ftp_para_reuse->task_id[i][j];
         if(model->ftp_para_reuse->schedule[task] == 1){
            dw1 = model->ftp_para_reuse->input_tiles[task][0].w1;
            dw2 = model->ftp_para_reuse->input_tiles[task][0].w2;
            dh1 = model->ftp_para_reuse->input_tiles[task][0].h1;
            dh2 = model->ftp_para_reuse->input_tiles[task][0].h2;
            // extra storage
            model->ftp_para_reuse->shrinked_input[task] = 
                                  crop_feature_maps(get_model_input(model), 
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].w, 
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].h,
                                  net_para->input_maps[model->ftp_para_reuse->from_layer].c, 
                                  dw1, dw2, dh1, dh2);
            model->ftp_para_reuse->shrinked_input_size[task] = 
                          sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->input_maps[model->ftp_para_reuse->from_layer].c;

#if DEBUG_FTP
            fprintf(stderr, "ck task id %d: dw1 %d, dw2 %d, dh1 %d, dh2 %d, f %u, size %d\n", task, dw1, dw2, dh1, dh2, model->ftp_para_reuse->from_layer, sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net_para->input_maps[model->ftp_para_reuse->from_layer].c);
#endif
         }
      }
   }
#endif

}

// for local source device
blob* local_dequeue_and_merge(device_ctxt* ctxt){
   /*Check if there is a data frame whose tasks have all been collected*/
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* temp = dequeue(ctxt->ready_pool);
#if DEBUG_FLAG
   printf("Check ready_pool... : Client %d is ready, merging the results\n", temp->id);
#endif
   uint32_t cli_id = temp->id;
   free_blob(temp);

   ftp_parameters *ftp_para = model->ftp_para;
   network_parameters *net_para = model->net_para;

   uint32_t stage_outs =  (net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w)*(net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h)*(net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c);
   float* stage_out = (float*) malloc(sizeof(float)*stage_outs);  
   uint32_t stage_out_size = sizeof(float)*stage_outs;  
   uint32_t part = 0;
   uint32_t task = 0;
   uint32_t frame_num = 0;
   uint32_t sp_id = 0;
   float* cropped_output;

   for(part = 0; part < ftp_para->partitions; part ++){
      temp = dequeue(ctxt->ready_queue);
      task = get_blob_task_id(temp);
      frame_num = get_blob_frame_seq(temp);
      sp_id = get_blob_sp_id(temp);

      if(net_para->type[ftp_para->from_layer+ftp_para->fused_layers-1] == CONV_LAYER){
         tile_region tmp = relative_offsets(ftp_para->input_tiles[task][ftp_para->fused_layers-1], 
                                       ftp_para->output_tiles[task][ftp_para->fused_layers-1]);  
         cropped_output = crop_feature_maps((float*)temp->data, 
                      ftp_para->input_tiles[task][ftp_para->fused_layers-1].w, 
                      ftp_para->input_tiles[task][ftp_para->fused_layers-1].h, 
                      net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c, 
                      tmp.w1, tmp.w2, tmp.h1, tmp.h2);
      }else{cropped_output = (float*)temp->data;}

      stitch_feature_maps(cropped_output, stage_out, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w2,
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h2);

      if(net_para->type[ftp_para->from_layer+ftp_para->fused_layers-1] == CONV_LAYER){free(cropped_output);}

      free_blob(temp);
   }

   temp = new_blob_and_copy_data(cli_id, stage_out_size, (uint8_t*)stage_out);
   free(stage_out);
   annotate_blob(temp, cli_id, frame_num, task, model->cur_sp);
   return temp;
}

blob* dequeue_and_merge(device_ctxt* ctxt){
   /*Check if there is a data frame whose tasks have all been collected*/
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* temp = dequeue(ctxt->ready_pool);
#if DEBUG_FLAG
   printf("Check ready_pool... : Client %d is ready, merging the results\n", temp->id);
#endif
   uint32_t cli_id = temp->id;
   free_blob(temp);

   ftp_parameters *ftp_para = model->ftp_para;
   network_parameters *net_para = model->net_para;

   uint32_t stage_outs =  (net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w)*(net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h)*(net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c);
   float* stage_out = (float*) malloc(sizeof(float)*stage_outs);  
   uint32_t stage_out_size = sizeof(float)*stage_outs;  
   uint32_t part = 0;
   uint32_t task = 0;
   uint32_t frame_num = 0;
   float* cropped_output;

   for(part = 0; part < ftp_para->partitions; part ++){
      temp = dequeue(ctxt->results_pool[cli_id]);
      task = get_blob_task_id(temp);
      frame_num = get_blob_frame_seq(temp);

      if(net_para->type[ftp_para->from_layer+ftp_para->fused_layers-1] == CONV_LAYER){
         tile_region tmp = relative_offsets(ftp_para->input_tiles[task][ftp_para->fused_layers-1], 
                                       ftp_para->output_tiles[task][ftp_para->fused_layers-1]);  
         cropped_output = crop_feature_maps((float*)temp->data, 
                      ftp_para->input_tiles[task][ftp_para->fused_layers-1].w, 
                      ftp_para->input_tiles[task][ftp_para->fused_layers-1].h, 
                      net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c, 
                      tmp.w1, tmp.w2, tmp.h1, tmp.h2);
      }else{cropped_output = (float*)temp->data;}

      stitch_feature_maps(cropped_output, stage_out, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].w, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].h, 
                          net_para->output_maps[ftp_para->from_layer+ftp_para->fused_layers-1].c, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w2,
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h2);

      if(net_para->type[ftp_para->from_layer+ftp_para->fused_layers-1] == CONV_LAYER){free(cropped_output);}

      free_blob(temp);
   }

   temp = new_blob_and_copy_data(cli_id, stage_out_size, (uint8_t*)stage_out);
   free(stage_out);
   annotate_blob(temp, cli_id, frame_num, task, model->cur_sp+1);
   return temp;
}

