# Compiler and flags
CC      ?= gcc
CFLAGS  ?= -std=c99 -Wall -Wextra -O2 -march=native -D_GNU_SOURCE
LDFLAGS ?= -lm -D_GNU_SOURCE

# Debug build
ifeq ($(DEBUG),1)
CFLAGS += -g -DDEBUG -O0
endif

# SIMD
ifeq ($(SIMD),avx2)
CFLAGS += -mavx2 -mfma -DUSE_AVX2
endif

# CUDA backend
ifeq ($(CUDA),1)
NVCC      ?= nvcc
NVCCFLAGS ?= -O2 -arch=sm_75 --compiler-options -fPIC -DUSE_CUDA
CFLAGS    += -DUSE_CUDA
CUDA_OBJS  = src/cuda_backend.o
LDFLAGS   += -lcublas -lcudart
endif

# Sources
SRC_DIR = src
SRCS    = $(wildcard $(SRC_DIR)/*.c)
OBJS    = $(SRCS:.c=.o)
TARGET  = lfmc

# Tests
TEST_DIR  = tests
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:.c=)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

src/cuda_backend.o: src/cuda_backend.cu src/cuda_backend.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "==> $$t"; ./$$t; done

$(TEST_DIR)/%: $(TEST_DIR)/%.c $(filter-out $(SRC_DIR)/main.o, $(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_BINS) $(CUDA_OBJS) src/cuda_backend.o
