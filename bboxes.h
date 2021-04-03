#pragma once

#include <vectors.h>

template <class T>
struct BBox3
{
    rtvector<T, 3> vmin, vmax;
};

typedef BBox3<float> BBox3f;