build: all

IFLAGS = 
CFLAGS =
LFLAGS = -libverbs -lrdmacm

BUILD_DIR := build

# MLU = /usr/local/neuware
# ROCM = /usr/local/neuware

ifndef MLU
ifndef ROCM
	CFLAGS += -DHAVE_HOST_RUNTIME=1
	CC := clang
endif
endif

ifdef MLU
	CC := clang

	IFLAGS += -I/usr/local/neuware/include

	CFLAGS += -DNEUWARE_HOME=/usr/local/neuware
	CFLAGS += -DHAVE_MLU_RUNTIME=1

	LFLAGS += -lcnrt -lmluops -lcndrv
	LFLAGS += -L/usr/local/neuware/lib64 
	LFLAGS += -L/usr/local/neuware/lib
endif

ifdef ROCM
	CC := hipcc

	IFLAGS += -I/opt/dtk/include/hip
	IFLAGS += -I/opt/dtk/include/hsa

	CFLAGS += -DHAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF=1
	CFLAGS += -DHAVE_ROCM_RUNTIME=1

	LFLAGS += -L/usr/local/dtk/lib
	LFLAGS += -L/usr/local/dtk/lib64
endif

all:
	@mkdir -p $(BUILD_DIR)

	${CC} ${IFLAGS} ${CFLAGS} -c rdma_client.c -o $(BUILD_DIR)/rdma_client.o
	${CC} ${IFLAGS} ${CFLAGS} -c rdma_server.c -o $(BUILD_DIR)/rdma_server.o
	${CC} ${IFLAGS} ${CFLAGS} -c common.c -o $(BUILD_DIR)/common.o

	${CC} ${IFLAGS} ${CFLAGS} ${LFLAGS} $(BUILD_DIR)/common.o $(BUILD_DIR)/rdma_client.o -o $(BUILD_DIR)/rdma_client
	${CC} ${IFLAGS} ${CFLAGS} ${LFLAGS} $(BUILD_DIR)/common.o $(BUILD_DIR)/rdma_server.o -o $(BUILD_DIR)/rdma_server

clean:
	rm -r $(BUILD_DIR)/*.o
	rm $(BUILD_DIR)/rdma_client $(BUILD_DIR)/rdma_server