#include "common.h"

void mem_alloc(mem_t *buf){
#if HAVE_HOST_RUNTIME
    host_buffer_malloc(buf);
#endif
#if HAVE_MLU_RUNTIME
    mlu_buffer_malloc(buf);
#endif
#if HAVE_ROCM_RUNTIME
    rocm_buffer_malloc(buf);
#endif
}

void mem_free(mem_t *buf){
#if HAVE_HOST_RUNTIME
    host_buffer_free(buf);
#endif
#if HAVE_MLU_RUNTIME
    mlu_buffer_free(buf);
#endif
#if HAVE_ROCM_RUNTIME
    rocm_buffer_free(buf);
#endif
}

#if HAVE_HOST_RUNTIME
void host_buffer_malloc(mem_t *buf) {
    buf->ptr = calloc(1, buf->size);
    rdma_info("host_buffer_malloc: %p\n", buf->ptr);
	if (!buf) {
		rdma_error("failed to allocate host buffer\n");
		return;
	}
    return;
}

void host_buffer_free(mem_t *buf) {
    if (buf->ptr) {
		free(buf->ptr);
	}
}
#endif

#if HAVE_MLU_RUNTIME
void mlu_init(void)
{
    static pthread_mutex_t cnnl_init_mutex = PTHREAD_MUTEX_INITIALIZER;
    CNresult ret;
    
    // cnInit Driver
    ret = cnInit(0);
    if (ret != CN_SUCCESS) {
        rdma_error("failed to cnInit %d", ret);
        return STATUS_ERROR;
    }

    // need create context first
    CNcontext context;
	ret = cnCtxCreate(&context, 0, 0);
	if (ret != CN_SUCCESS) {
        rdma_error("failed to create cnCtx %d.", ret);
        return -1;
    }

    gpu_initialized = 1;

end:
    pthread_mutex_unlock(&cnnl_init_mutex);
}

void mlu_buffer_malloc(mem_t *buf) {
    CNresult ret;
    ret = cnMallocPeerAble(&(buf->addr), buf->size);

	if (ret == CN_ERROR_NOT_INITIALIZED) {
        rdma_error("CNDrv has not been initialized with cnInit or CNDrv fails to be initialized.");
    }
    if (ret == CN_ERROR_INVALID_VALUE) {
        rdma_error("The parameters passed to this API are not within an acceptable value range.");
    }
    if (ret != CN_SUCCESS) {
        rdma_error("failed to allocate memory %d.", ret);
    }

    buf->ptr = (void *)buf->addr;
}

void mlu_buffer_free(mem_t *buf) {
    CNresult ret;
    ret = cnFree(buf->addr);
    if (ret != CN_SUCCESS) {
        rdma_error("failed to free mlu memory");
    }
    if (buf->ptr) {
		free(buf->ptr);
	}
}
#endif

#if HAVE_ROCM_RUNTIME
void rocm_init(void)
{
    static pthread_mutex_t rocm_init_mutex = PTHREAD_MUTEX_INITIALIZER;
    hsa_status_t hsa_status;

    if (pthread_mutex_lock(&rocm_init_mutex) == 0) {
        if (gpu_initialized) {
            goto end;
        }
    } else  {
        rdma_error("Could not take mutex");
    }

    memset(&rocm_agents, 0, sizeof(rocm_agents));

    hsa_status = hsa_init();
    if (hsa_status != HSA_STATUS_SUCCESS) {
        rdma_error("Failure to open HSA connection: 0x%x", hsa_status);
        goto end;
    }

#if ROCM_DMABUF_SUPPERTED
    hsa_status = rocmLibraryInit();
    if (hsa_status != STATUS_SUCCESS) {
        rdma_error("Failure to initialize ROCm library: 0x%x", hsa_status);
        goto end;
    }
#endif

    gpu_initialized = 1;

end:
    pthread_mutex_unlock(&rocm_init_mutex);
}

void rocm_buffer_malloc(mem_t *buf) {
    hipError_t ret;
    ret = hipMalloc(&(buf->ptr), buf->size);
    if (ret != hipSuccess) {
        rdma_error("failed to allocate rocm memory");
    }
}

void rocm_buffer_free(mem_t *buf) {
    
    hipError_t ret;
    ret = hipFree(&(buf->ptr));
    if (ret != hipSuccess) {
        rdma_error("failed to free rocm memory");
    }
}
#endif