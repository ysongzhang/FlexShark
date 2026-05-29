#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/u8.hpp>

namespace shark
{
    namespace protocols
    {
        namespace output
        {
            void gen(const shark::span<u64> &X);
            void gen(const shark::span<u32> &X);
            void gen(const shark::span<u16> &X);
            void gen(const shark::span<u8> &X);
            void eval(shark::span<u64> &X);
            void eval(shark::span<u32> &X);
            void eval(shark::span<u16> &X);
            void eval(shark::span<u8> &X);
            void call(shark::span<u64> &X);
            void call(shark::span<u32> &X);
            void call(shark::span<u16> &X);
            void call(shark::span<u8> &X);
        }
    }
}
