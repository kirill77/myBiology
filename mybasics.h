#pragma once

typedef unsigned int NvU32;

#ifdef NDEBUG
#define ASSERT_ONLY_CODE 0
#define nvAssert(x)
#else
#define ASSERT_ONLY_CODE 1
#define nvAssert(x) if (!x) { __debugbreak(); }
#endif