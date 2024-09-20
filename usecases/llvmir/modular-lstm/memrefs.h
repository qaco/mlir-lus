#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <stdint.h>

#include "error.h"

#ifndef MEMREFS_H
#define MEMREFS_H

struct memref_2d_f32 {
  float* allocated ;
  float* aligned ;
  intptr_t offset ;
  intptr_t size[2] ;
  intptr_t stride[2] ;
};

void alloc_memref_2d_f32(struct memref_2d_f32*s, int size0, int size1);
void free_memref_2d_f32(struct memref_2d_f32*s);
void print_memref_2d_f32(struct memref_2d_f32*s);
void bzero_memref_2d_f32(struct memref_2d_f32*s);

#endif
