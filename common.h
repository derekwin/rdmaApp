#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#if HAVE_MLU_RUNTIME
#include "cnrt.h"
#include "mlu_op.h"
#include "cn_api.h" // CNresult
#include <unistd.h> // getpid()
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
static inline const char *getErrorName(CNresult error)
{
    const char *str;
    cnGetErrorName(error, &str);
    return str;
}

static inline const char *getErrorString(CNresult error)
{
    const char *str;
    cnGetErrorString(error, &str);
    return str;
}
#define ERROR_CHECK(ret)                                                                           \
     do {                                                                                           \
         CNresult r__ = (ret);                                                                      \
         if (r__ != CN_SUCCESS) {                                                                   \
             printf(                                                                                \
                 "error occur, func: %s, line: %d, ret:%d, cn_error_code:%s, cn_error_string:%s\n", \
                 __func__, __LINE__, r__, getErrorName(r__), getErrorString(r__));                  \
             return;                                                                               \
         }                                                                                          \
     } while (0)

static CNdev cnDevice;
static CNcontext cnContext;

void mlu_init();
void mlu_buffer_malloc(mem_t *buf);
void mlu_buffer_free(mem_t *buf);
#endif

#if HAVE_ROCM_RUNTIME
#define MAX_AGENTS 127
static struct agents {
    int num;
    hsa_agent_t agents[MAX_AGENTS];
    int num_gpu;
    hsa_agent_t gpu_agents[MAX_AGENTS];
} rocm_agents;
void rocm_init();
void rocm_buffer_malloc(mem_t *buf);
void rocm_buffer_free(mem_t *buf);
#endif