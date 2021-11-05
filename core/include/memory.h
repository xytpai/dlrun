#pragma once

#include "dlrun.h"
#include "device_schema.h"

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

namespace dl {

template <typename scalar_t>
void empty_(std::string name, std::vector<int> shape, int device) {
    Tensor t;
    t.dtype = TYPE(scalar_t);
    t.shape = shape;
    t.numel = 1;
    for (auto d : shape) t.numel *= d;
    if (device < 0) {
        t.data = MALLOC_CPU(t.numel * sizeof(scalar_t));
    } else {
        t.data = MALLOC_XPU(device, t.numel * sizeof(scalar_t));
    }
    GLOBAL_TENSOR[name] = t;
}

void empty(Params &p) {
    HANDLE_DTYPE2(int, float, p.s["dtype"], empty_, p.s["name"], p.vi["shape"], p.i["device"])
}

} // namespace dl