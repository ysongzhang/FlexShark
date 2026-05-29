#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>

namespace shark {
    namespace protocols {
        namespace mul {
            void gen(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void gen(const shark::span<u32> &X, const shark::span<u32> &Y, shark::span<u32> &Z);
            void eval(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void eval(const shark::span<u32> &X, const shark::span<u32> &Y, shark::span<u32> &Z);
            void call(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void call(const shark::span<u32> &X, const shark::span<u32> &Y, shark::span<u32> &Z);
            shark::span<u64> call(const shark::span<u64> &X, const shark::span<u64> &Y);
            shark::span<u32> call(const shark::span<u32> &X, const shark::span<u32> &Y);
        }
    }
}