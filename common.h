#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#if HAVE_MLU_RUNTIME
#include "cnrt.h"
#include "mlu_op.h"
#include "cn_api.h" // CNresult
#endif
#if HAVE_ROCM_RUNTIME
#include <hsa.h>
#include <hip/hip_runtime.h>
// hipError_t hipSuccess
#include <hsa_ext_amd.h>
#endif

/* Error Macro*/
#define rdma_error(msg, args...) do {\
	fprintf(stderr, "%s : %d : ERROR : " msg, __FILE__, __LINE__, ## args);\
}while(0);
/* Error Macro*/
#define rdma_info(msg, args...) do {\
	fprintf(stderr, "%s : %d : INFO : " msg, __FILE__, __LINE__, ## args);\
}while(0);

static int gpu_initialized = 0;

typedef struct mem_s {
  void *ptr;
#if HAVE_MLU_RUNTIME
  CNaddr addr;
#endif
  int size;
} mem_t;

void mem_alloc(mem_t *buf);
void mem_free(mem_t *buf);

#if HAVE_HOST_RUNTIME
void host_buffer_malloc(mem_t *buf);
void host_buffer_free(mem_t *buf);
#endif

#if HAVE_MLU_RUNTIME
void mlu_init();
void mlu_buffer_malloc(mem_t *buf);
void mlu_buffer_free(mem_t *buf);
#endif

#if HAVE_ROCM_RUNTIME
void rocm_init();
void rocm_buffer_malloc(mem_t *buf);
void rocm_buffer_free(mem_t *buf);
#endif