#ifndef SCHEDULE_H
#define SCHEDULE_H
#include <stdint.h>

// TODO(lizhou): update function to return by ratio.
int32_t get_task_dst_id(uint32_t task, uint32_t total_tasks, uint32_t num_devices);

#endif
