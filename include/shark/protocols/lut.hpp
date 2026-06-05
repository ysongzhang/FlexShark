#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace lut
        {
            void gen(const shark::span<u64> &r_X, shark::span<u64> &r_Y, const std::vector<u64> &lut, int bin);
            void gen(const shark::span<u32> &r_X, shark::span<u32> &r_Y, const std::vector<u32> &lut, int bin);
            void gen(const shark::span<u16> &r_X, shark::span<u16> &r_Y, const std::vector<u16> &lut, int bin);

            void eval(const shark::span<u64> &X, shark::span<u64> &Y, const std::vector<u64> &lut, int bin);
            void eval(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut, int bin);
            void eval(const shark::span<u16> &X, shark::span<u16> &Y, const std::vector<u16> &lut, int bin);

            void call(const shark::span<u64> &X, shark::span<u64> &Y, const std::vector<u64> &lut, int bin);
            void call(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut, int bin);
            void call(const shark::span<u16> &X, shark::span<u16> &Y, const std::vector<u16> &lut, int bin);

            shark::span<u64> call(const shark::span<u64> &X, const std::vector<u64> &lut, int bin);
            shark::span<u32> call(const shark::span<u32> &X, const std::vector<u32> &lut, int bin);
            shark::span<u16> call(const shark::span<u16> &X, const std::vector<u16> &lut, int bin);

            void gen(const shark::span<u32> &r_X, shark::span<u32> &r_Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin);
            void eval(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin);
            void call(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin);
            shark::span<u32> call(const shark::span<u32> &X, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin);
        }

    }
}
