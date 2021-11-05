#include "dlrun.h"

int main() {
    using namespace std;
    using namespace dl;

    REGISTER(empty, "empty");
    REGISTER(linspace_, "linspace_");

    Params p;
    p.s["name"] = "Tensor0";
    p.s["dtype"] = TYPE(float);
    p.vi["shape"] = std::vector<int>{2, 2};
    EXEC("empty", p);

    Params p2;
    p2.s["name"] = "Tensor0";
    EXEC("linspace_", p2);

    Tensor &t = GLOBAL_TENSOR["Tensor0"];
    float *ptr = (float *)t.data;
    for (int i = 0; i < t.numel; i++) {
        cout << ptr[i] << endl;
    }
}