#include "schedule.h"

int32_t get_task_dst_id(uint32_t task, uint32_t total_tasks, uint32_t num_devices) {
  // evenly assign
  //return task / (total_tasks / num_devices);
  int32_t a = total_tasks / num_devices;
  int32_t b = total_tasks % num_devices;
  uint32_t cur = 0;
  for (int32_t i = 0; i < num_devices; i++) {
    uint32_t next = cur + a + (b > 0 ? 1 : 0); 
    if (task >= cur && task < next) return i;
    cur = next;
    b--;
  }

  return -1;
}
