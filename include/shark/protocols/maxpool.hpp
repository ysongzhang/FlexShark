#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace maxpool
        {
            void call(u64 f, u64 padding, u64 stride, u64 ci, u64 inH, u64 inW, const shark::span<u64> &Img, shark::span<u64> &OutImg);
            shark::span<u64> call(u64 f, u64 padding, u64 stride, u64 ci, u64 inH, u64 inW, const shark::span<u64> &Img);

            // batch max computation
            void max(u64 s1, u64 s2, const shark::span<u32> &in, shark::span<u32> &out);
        }

    }
}
