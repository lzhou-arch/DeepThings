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

// modified to c

#include <unistd.h> // usleep
#include <stdio.h>  // FILE
#include <stdlib.h> // atoi
#include <string.h>

#include "cpu.h"

//uint8_t get_cpu_count()
//{
//  return sysconf( _SC_NPROCESSORS_ONLN );
//}

float cpu_percentage(unsigned cpu_usage_delay) {
  
  // cpu stats
  // user, nice, system, idle
  // in that order
  unsigned long long stats[CP_STATES];

  FILE * inf = fopen("/proc/stat", "r");
  char line[1024];
  // skip first line of overal cpu
  fgets(line, 1024, inf);
  // read cpu0
  fgets(line, 1024, inf);
  char delims[] = "\t, ";
  char * t;
  // skip "cpu" 
  t = strtok(line, delims);
  // parse cpu line
  for( unsigned i=0; i < 4; i++ ) {
    t = strtok(NULL, delims);
    stats[i] = atoi(t);
  }
  fclose(inf);

  usleep(cpu_usage_delay);

  inf = fopen("/proc/stat", "r");
  // skip first line of overal cpu
  fgets(line, 1024, inf);
  // read cpu0
  fgets(line, 1024, inf);
  // skip "cpu" 
  t = strtok(line, delims);
  // parse cpu line
  for( unsigned i=0; i < 4; i++ ) {
    t = strtok(NULL, delims);
    stats[i] = atoi(t) - stats[i];
  }
  fclose(inf);

  return (float)( 
    stats[CP_USER] + stats[CP_NICE] + stats[CP_SYS]
    ) / (float)( 
        stats[CP_USER] + stats[CP_NICE] + stats[CP_SYS] + stats[CP_IDLE] 
    ) * 100.0;
}

//int main() {
//  printf("cpu usage:\n");
//  while(1) {
//    printf("%lf\n", cpu_percentage(20000)); // > 20ms
//  }
//}
