/* vim: tabstop=2 shiftwidth=2 expandtab textwidth=80 linebreak wrap
 *
 * Copyright 2012 Matthew McCormick
 * Copyright 2015 Pawel 'l0ner' Soltys
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>  // FILE
#include <stdlib.h> // atoi
#include <string.h>
#include <sys/sysinfo.h>

#include "memory.h"

void mem_status( MemoryStatus * status )
{
  FILE * inf = fopen("/proc/meminfo", "r");
  char line[1024];

  unsigned int total_mem;
  unsigned int used_mem;

  /* Since linux uses some RAM for disk caching, the actuall used ram is lower
   * than what sysinfo(), top or free reports. htop reports the usage in a
   * correct way. The memory used for caching doesn't count as used, since it
   * can be freed in any moment. Usually it hapens automatically, when an
   * application requests memory.
   * In order to calculate the ram that's actually used we need to use the
   * following formula:
   *    total_ram + shmem - free_ram - buffered_ram - cached_ram - srclaimable
   *
   * example data, junk removed, with comments added:
   *
   * MemTotal:        61768 kB    old
   * MemFree:          1436 kB    old
   * MemAvailable     ????? kB    ??
   * MemShared:           0 kB    old (now always zero; not calculated)
   * Buffers:          1312 kB    old
   * Cached:          20932 kB    old
   * SwapTotal:      122580 kB    old
   * SwapFree:        60352 kB    old
   */

  //std::ifstream memory_info("/proc/meminfo");

  char delims[] = "\t,: ";
  char * t;
  while( fgets(line, 1024, inf) != NULL ) {
    t = strtok(line, delims);

    if( strcmp(t, "MemTotal") == 0 )
    {
      // get total memory
      total_mem = atoi( strtok(NULL, delims) );
    }
    else if( strcmp(t, "MemFree") == 0 )
    {
      used_mem = total_mem - atoi( strtok(NULL, delims) );
    }
    else if( strcmp(t, "Shmem") == 0 )
    {
      used_mem += atoi( strtok(NULL, delims) );
    }
    else if( strcmp(t, "Buffers") == 0 ||
             strcmp(t, "Cached") == 0  ||
             strcmp(t, "SReclaimable") == 0 )
    {
      used_mem -= atoi( strtok(NULL, delims) );
    }
  }
  fclose(inf);

  status->used_mem = (float)( used_mem );
  status->total_mem = (float)( total_mem );
}

//int main() {
//  struct MemoryStatus status;
//  printf("Mem usage:\n");
//  mem_status(&status);
//  printf("%f/%f (%f)\n", status.used_mem, status.total_mem,
//      status.used_mem/status.total_mem);
//
//  return 0;
//}
