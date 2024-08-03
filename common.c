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
    
    int mlu_pci_bus_id;
	int mlu_pci_device_id;
	int index;
    int deviceCount = 0;
    int mlu_device_id;
	CNdev cn_device;
    CNresult error;
    char name[128];
    int group_index = 0; // choose first device

	printf("initializing MLU\n");
	error = cnInit(0);
	if (error != CN_SUCCESS) {
		printf("cnInit(0) returned %d\n", error);
		return; 
	}

    error = cnDeviceGetCount(&deviceCount);
	if (error != CN_SUCCESS) {
		printf("cnDeviceGetCount() returned %d\n", error);
		return; 
	}
    if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		return;
    }

    mlu_device_id = group_index % deviceCount;
	
    if (mlu_device_id >= deviceCount) {
		fprintf(stderr, "No such device ID (%d) exists in system\n", mlu_device_id);
		return;
	}

	printf("Listing all MLU devices in system:\n");
	for (index = 0; index < deviceCount; index++) {
		ERROR_CHECK(cnDeviceGet(&cn_device, index));
		cnDeviceGetAttribute(&mlu_pci_bus_id, CN_DEVICE_ATTRIBUTE_PCI_BUS_ID , cn_device);
		cnDeviceGetAttribute(&mlu_pci_device_id, CN_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cn_device);
		printf("MLU device %d: PCIe address is %02X:%02X\n", index, (unsigned int)mlu_pci_bus_id, (unsigned int)mlu_pci_device_id);
	}

	printf("\nPicking device No. %d\n", mlu_device_id);

    ERROR_CHECK(cnDeviceGet(&cnDevice, mlu_device_id));

	ERROR_CHECK(cnDeviceGetName(name, sizeof(name), mlu_device_id));
	printf("[pid = %d, dev = %ld] device name = [%s]\n", getpid(), cnDevice, name);
	printf("creating MLU Ctx\n");

	/* Create context */
	error = cnCtxCreate(&cnContext, 0, cnDevice);
	if (error != CN_SUCCESS) {
		printf("cnCtxCreate() error=%d\n", error);
		return;
	}

	printf("making it the current CUDA Ctx\n");
	error = cnCtxSetCurrent(cnContext);
	if (error != CN_SUCCESS) {
		printf("cnCtxSetCurrent() error=%d\n", error);
		return;
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
    ERROR_CHECK(cnCtxDestroy(cnContext));
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