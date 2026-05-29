#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace up
        {
            void gen(const shark::span<u32> &r_in, shark::span<u64> &r_out, int f, int f_prime);
            void gen(const shark::span<u16> &r_in, shark::span<u64> &r_out, int f, int f_prime);
            void gen(const shark::span<u16> &r_in, shark::span<u32> &r_out, int f, int f_prime);

            void eval(const shark::span<u32> &in, shark::span<u64> &out, int f, int f_prime);
            void eval(const shark::span<u16> &in, shark::span<u64> &out, int f, int f_prime);
            void eval(const shark::span<u16> &in, shark::span<u32> &out, int f, int f_prime);

            void call(const shark::span<u32> &in, shark::span<u64> &out, int f, int f_prime);
            void call(const shark::span<u16> &in, shark::span<u64> &out, int f, int f_prime);
            void call(const shark::span<u16> &in, shark::span<u32> &out, int f, int f_prime);

            shark::span<u64> call_32_64(const shark::span<u32> &in, int f, int f_prime);
            shark::span<u64> call_16_64(const shark::span<u16> &in, int f, int f_prime);
            shark::span<u32> call_16_32(const shark::span<u16> &in, int f, int f_prime);
        }

    }
}