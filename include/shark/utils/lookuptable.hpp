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

    inline u32 ceil_log2_u32(u32 x) {
        if (x <= 1) return 1;
        // floor(log2(x))
        u32 l = 31 - __builtin_clz(x);
        if ((x & (x - 1)) == 0)
            return l;
        return l + 1;
    }

    void init_luts();


//     // Init inv
//     constexpr u32 table_size_inv = (u32)ceil((1.0  - 1.0 / SCALE_BASE) * (1u << FLOAT_PRECISION_32));
//     constexpr u32 pow2_bit_inv = ceil_log2_u32_constexpr(table_size_inv);
//     constexpr u32 lut_size_inv = 1u << pow2_bit_inv;
//     // constexpr std::array<u32, lut_size_inv> make_lut_inv()
//     // {
//     //     std::array<u32, lut_size_inv> lut{};

//     //     for (u32 i = 0; i < table_size_inv; ++i)
//     //     {
//     //         float real_value = float(i) / (1u << FLOAT_PRECISION_32) + 1.0 / SCALE_BASE;
//     //         lut[i] = u32((1.0 / real_value) * (1u << FLOAT_PRECISION_32));
//     //     }
//     //     return lut;
//     // }
//     // inline const auto lut_inv_array = make_lut_inv();

//     // Init exp_2
//     constexpr u32 table_size_exp_2 = FLOAT_PRECISION_32 + 2;
//     constexpr u32 pow2_bit_exp_2 = ceil_log2_u32_constexpr(table_size_exp_2);
//     constexpr u32 lut_size_exp_2 = 1u << pow2_bit_exp_2;


//     // Init exp_b
//     constexpr u32 table_size_exp_b = u32((int)(log(pow(2, 2 * FLOAT_PRECISION_32)) / log(SCALE_BASE)) + 1);
//     constexpr u32 pow2_bit_exp_b = ceil_log2_u32_constexpr(table_size_exp_b);
//     constexpr u32 lut_size_exp_b = 1u << pow2_bit_exp_b;
}
