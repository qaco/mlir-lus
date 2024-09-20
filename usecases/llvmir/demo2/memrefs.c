#include "memrefs.h"

void alloc_memref_2d_f32(struct memref_2d_f32*s,
			 int size0, int size1) {
  int size = size0*size1*sizeof(float) ;
  s->allocated = malloc(size+sizeof(float)) ;
  uintptr_t offset = ((uintptr_t)s->allocated)%((uintptr_t)sizeof(float)) ;
  if(offset>0) {
    offset = ((uintptr_t)sizeof(float))-offset ;
    s->allocated = (void*)(((uintptr_t)s->allocated)+offset) ;
  }
  s->aligned = s->allocated ;
  s->offset = 0 ;
  s->size[0] = size0 ;
  s->size[1] = size1 ;
  s->stride[0] = 1 ;
  s->stride[1] = 1 ;
}

void free_memref_2d_f32(struct memref_2d_f32*s) {
  free(s->allocated) ;
}

void print_memref_2d_f32(struct memref_2d_f32*s) {
  printf("memref<%ldx%ldxf32>:\n",s->size[0],s->size[1]) ;
  for(int i=0;i<s->size[0];i++) {
    printf("\t") ;
    for(int j=0;j<s->size[1];j++)
      printf("%f ",s->aligned[i*s->size[1]+j]) ;
    printf("\n") ;
  }
  fflush(stdout) ;
}

void bzero_memref_2d_f32(struct memref_2d_f32*s) {
  bzero(s->aligned,s->size[0]*s->size[1]*sizeof(float)) ;
}
