#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "_main.h"

int main(int argc, char** argv) {
  int step_c;
  int step_max;
  float data[224][224][3];
  Resnet__resnet50_out _res;
  step_c = 0;
  step_max = 0;
  if ((argc==2)) {
    step_max = atoi(argv[1]);
  };
  struct timeval time;
  gettimeofday(&time, NULL);
  unsigned long start = time.tv_sec;
  unsigned long start_usec = time.tv_usec;
  while ((!(step_max)||(step_c<step_max))) {
    step_c = (step_c+1);
    Resnet__resnet50_step(data, &_res);
    gettimeofday(&time, NULL);
    unsigned long stop = time.tv_sec;
    unsigned long stop_usec = time.tv_usec;
    unsigned long elapsed = (stop - start) * 1000000 + stop_usec - start_usec;
    printf("time since last tick: %lu us\n", elapsed);
    gettimeofday(&time, NULL);
    start = time.tv_sec;
    start_usec = time.tv_usec;
    fflush(stdout);
  };
  return 0;
}

