#pragma once

#include <cstdint>
#include <cmath>
#include <stdexcept>

constexpr int FLOAT_PRECISION_64 = 16;
constexpr int FLOAT_PRECISION_32 = 12;
constexpr int FLOAT_PRECISION_16 = 6;

#define SCALE_BASE 256


/*******************************
 *  Math Utilities
 *******************************/
#define ReLU(x) ((x) > 0 ? (x) : 0)
#define GeLU(x) (0.5f * (x) * (1 + tanh(sqrt(2.0f / M_PI) * ((x) + 0.044715f * pow((x), 3)))))