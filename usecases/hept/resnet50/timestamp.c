#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h> 

void timestamp() {
  static unsigned long start = 0;
  static unsigned long start_usec = 0;
  struct timeval time;
  gettimeofday(&time, NULL);
  if (start) {
    unsigned long stop = time.tv_sec;
    unsigned long stop_usec = time.tv_usec;
    unsigned long elapsed = (stop - start) * 1000000 + stop_usec - start_usec;
    printf("elapsed: %lu us\n", elapsed);
  }
  else {
    start = time.tv_sec;
    start_usec = time.tv_usec;
  }
}
