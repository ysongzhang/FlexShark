#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace down
        {
            void gen(const shark::span<u64> &r_in, shark::span<u32> &r_out, int f, int f_prime);
            void gen(const shark::span<u32> &r_in, shark::span<u16> &r_out, int f, int f_prime);
            void gen(const shark::span<u64> &r_in, shark::span<u16> &r_out, int f, int f_prime);

            void eval(const shark::span<u64> &in, shark::span<u32> &out, int f, int f_prime);
            void eval(const shark::span<u32> &in, shark::span<u16> &out, int f, int f_prime);
            void eval(const shark::span<u64> &in, shark::span<u16> &out, int f, int f_prime);

            void call(const shark::span<u64> &in, shark::span<u32> &out, int f, int f_prime);
            void call(const shark::span<u32> &in, shark::span<u16> &out, int f, int f_prime);
            void call(const shark::span<u64> &in, shark::span<u16> &out, int f, int f_prime);

            shark::span<u32> call_64_32(const shark::span<u64> &in, int f, int f_prime);
            shark::span<u16> call_32_16(const shark::span<u32> &in, int f, int f_prime);
            shark::span<u16> call_64_16(const shark::span<u64> &in, int f, int f_prime);
        }

    }
}