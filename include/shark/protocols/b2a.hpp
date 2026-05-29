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
        namespace b2a
        {
            void gen(const shark::span<u8> &s, shark::span<u16> &res);
            void gen(const shark::span<u8> &s, shark::span<u32> &res);
            void gen(const shark::span<u8> &s, shark::span<u64> &res);

            void eval(const shark::span<u8> &s, shark::span<u16> &res);
            void eval(const shark::span<u8> &s, shark::span<u32> &res);
            void eval(const shark::span<u8> &s, shark::span<u64> &res);

            void call(const shark::span<u8> &s, shark::span<u16> &res);
            void call(const shark::span<u8> &s, shark::span<u32> &res);
            void call(const shark::span<u8> &s, shark::span<u64> &res);

            shark::span<u64> call(const shark::span<u8> &s);
            shark::span<u32> call_32(const shark::span<u8> &s);
            shark::span<u16> call_16(const shark::span<u8> &s);
        }
    }
}
