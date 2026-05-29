#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace truncate
        {
            void gen(const shark::span<u64> &r_in, shark::span<u64> &r_out, int f);
            void gen(const shark::span<u32> &r_in, shark::span<u32> &r_out, int f);

            void eval(const shark::span<u64> &in, shark::span<u64> &out, int f);
            void eval(const shark::span<u32> &in, shark::span<u32> &out, int f);

            void call(const shark::span<u64> &in, shark::span<u64> &out, int f);
            void call(const shark::span<u32> &in, shark::span<u32> &out, int f);

            shark::span<u64> call(const shark::span<u64> &in, int f);
            shark::span<u32> call(const shark::span<u32> &in, int f);
        }

    }
}