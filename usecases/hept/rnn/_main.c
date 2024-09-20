/* --- Generated the 6/4/2021 at 14:59 --- */
/* --- heptagon compiler, version 1.05.00 (compiled thu. mar. 11 18:24:55 CET 2021) --- */
/* --- Command line: /home/hpompougnac/.opam/default/bin/heptc -target c -s rnn_example model.ept --- */

#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "_main.h"

Model__rnn_example_mem mem;
int main(int argc, char** argv) {
  int step_c;
  int step_max;
  float data[3][1];
  Model__rnn_example_out _res;
  step_c = 0;
  step_max = 0;
  if ((argc==2)) {
    step_max = atoi(argv[1]);
  };
  Model__rnn_example_reset(&mem);
  struct timeval time;
  gettimeofday(&time, NULL);
  unsigned long start = time.tv_sec;
  unsigned long start_usec = time.tv_usec;
  while ((!(step_max)||(step_c<step_max))) {
    step_c = (step_c+1);
    /* { */
    /*   int i; */
    /*   for (i = 0; i < 3; ++i) { */
    /*     { */
    /*       int i_22; */
    /*       for (i_22 = 0; i_22 < 1; ++i_22) { */
            
    /*         printf("data[%d][%d] ? ", i_22, i); */
    /*         scanf("%f", &data[i][i_22]);; */
    /*       } */
    /*     }; */
    /*   } */
    /* }; */
    Model__rnn_example_step(data, &_res, &mem);
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

