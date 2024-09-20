#ifndef MYMATH_H
#define MYMATH_H

#include <math.h>

typedef struct Mymath__sqrt_out {
  float o;
} Mymath__sqrt_out;

typedef struct Mymath__exp_out {
  float o;
} Mymath__exp_out;

typedef struct Mymath__int2float_out {
  float o;
} Mymath__int2float_out;

__attribute__((always_inline))
inline void Mymath__sqrt_step(float i, Mymath__sqrt_out* _out) {
  _out->o = sqrtf(i) ;
  return ;
}

__attribute__((always_inline))
inline void Mymath__exp_step(float i, Mymath__exp_out* _out) {
  _out->o = expf(i) ;
  return ;
}

__attribute__((always_inline))
inline void Mymath__int2float_step(int i, Mymath__int2float_out* _out) {
  _out->o = (float)i ;
  return ;
}

#endif
