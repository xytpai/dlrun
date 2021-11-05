#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

namespace dl {

struct Params {
    std::unordered_map<std::string, std::string> s;
    std::unordered_map<std::string, int> i;
    std::unordered_map<std::string, float> f;
    std::unordered_map<std::string, std::vector<int>> vi;
    std::unordered_map<std::string, std::vector<float>> vf;
};

struct Tensor {
    std::string dtype;
    std::vector<int> shape;
    void *data;
    int numel;
    int device;
};

std::unordered_map<std::string, void *> _func_d;
std::unordered_map<std::string, Tensor> _tensor_d;

#define REGISTER(FUNC, NAME) _func_d[NAME] = (void *)FUNC
#define EXEC(NAME, PARAMS) \
    (*((void (*)(Params &))_func_d[NAME]))(PARAMS)
#define TYPE(DATATYPE) typeid(DATATYPE).name()
#define GLOBAL_TENSOR _tensor_d
#define HANDLE_DTYPE(T, TYPENAME, FUNC, ...) \
    if (TYPE(T) == TYPENAME) {               \
        using scalar_t = T;                  \
        FUNC<scalar_t>(__VA_ARGS__);         \
    }
#define HANDLE_DTYPE2(T1, T2, ...) \
    HANDLE_DTYPE(T1, __VA_ARGS__)  \
    HANDLE_DTYPE(T2, __VA_ARGS__)
#define HANDLE_DTYPE3(T1, T2, T3, ...) \
    HANDLE_DTYPE(T1, __VA_ARGS__)      \
    HANDLE_DTYPE(T2, __VA_ARGS__)      \
    HANDLE_DTYPE(T3, __VA_ARGS__)
#define HANDLE_DTYPE4(T1, T2, T3, T4, ...) \
    HANDLE_DTYPE(T1, __VA_ARGS__)          \
    HANDLE_DTYPE(T2, __VA_ARGS__)          \
    HANDLE_DTYPE(T3, __VA_ARGS__)          \
    HANDLE_DTYPE(T4, __VA_ARGS__)

} // namespace dl

#include "device_schema.h"
#include "memory.h"
#include "operators.h"
