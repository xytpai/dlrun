#pragma once

#define MALLOC_CPU(SIZE) (void *)(new char[SIZE]);
#define DELETE_CPU(PTR) delete[] PTR;

#define MALLOC_XPU(DEVICE, SIZE) MALLOC_CPU(SIZE)

#ifdef USE_CUDA
#define MALLOC_XPU(DEVICE, SIZE) [=]() -> void * {  \
    cudaSetDevice(DEVICE);                          \
    void *ptr;                                      \
    cudaMalloc((void **)&ptr, SIZE * sizeof(char)); \
    return ptr;                                     \
}
}
#endif