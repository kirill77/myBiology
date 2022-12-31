#include "tensor.h"

TensorRef Tensor::cloneToPrecision(NvU32 elemSize)
{
    TensorRef r = std::make_shared<Tensor>(this->getDims(), elemSize);
    r->copySubregionFrom(0, *this, 0, size());
    return r;
}
