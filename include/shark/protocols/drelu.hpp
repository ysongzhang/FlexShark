#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/u8.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace drelu
        {
            void gen(const shark::span<u16> &X, shark::span<u8> &Y);
            void gen(const shark::span<u32> &X, shark::span<u8> &Y);
            void gen(const shark::span<u64> &X, shark::span<u8> &Y);

            void eval(const shark::span<u16> &X, shark::span<u8> &Y);
            void eval(const shark::span<u32> &X, shark::span<u8> &Y);
            void eval(const shark::span<u64> &X, shark::span<u8> &Y);

            void call(const shark::span<u16> &X, shark::span<u8> &Y);
            void call(const shark::span<u32> &X, shark::span<u8> &Y);
            void call(const shark::span<u64> &X, shark::span<u8> &Y);

            shark::span<u8> call(const shark::span<u16> &X);
            shark::span<u8> call(const shark::span<u32> &X);
            shark::span<u8> call(const shark::span<u64> &X);
        }
    }
}
