#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/utils/globals.hpp>


namespace shark
{
    extern std::vector<u32> lut_inv;
    extern int bin_inv;
    extern std::vector<u32> lut_exp_2;
    extern int bin_exp_2;
    extern std::vector<u32> lut_exp_b;
    extern int bin_exp_b;
    extern std::vector<u32> lut_rsqrt;
    extern int bin_rsqrt;
    extern std::vector<u32> lut_exsqrt;
    extern int bin_exsqrt;
    extern std::vector<u16> lut_gelu;
    extern int bin_gelu;
    extern std::vector<u32> lut_gelu_32;
    extern int bin_gelu_32;

    inline u32 ceil_log2_u32(u32 x) {
        if (x <= 1) return 1;
        // floor(log2(x))
        u32 l = 31 - __builtin_clz(x);
        if ((x & (x - 1)) == 0)
            return l;
        return l + 1;
    }

    void init_luts();
}
