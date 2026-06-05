#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace relu
        {
            void gen(const shark::span<u64> &r_X, shark::span<u64> &r_Y);
            void gen(const shark::span<u32> &r_X, shark::span<u32> &r_Y);
            void gen(const shark::span<u16> &r_X, shark::span<u16> &r_Y);
            void eval(const shark::span<u64> &X, shark::span<u64> &Y);
            void eval(const shark::span<u32> &X, shark::span<u32> &Y);
            void eval(const shark::span<u16> &X, shark::span<u16> &Y);
            void call(const shark::span<u64> &X, shark::span<u64> &Y);
            void call(const shark::span<u32> &X, shark::span<u32> &Y);
            void call(const shark::span<u16> &X, shark::span<u16> &Y);
            shark::span<u64> call(const shark::span<u64> &X);
            shark::span<u32> call(const shark::span<u32> &X);
            shark::span<u16> call(const shark::span<u16> &X);
        }

    }
}
