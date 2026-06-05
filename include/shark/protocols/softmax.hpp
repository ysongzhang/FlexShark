#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/maxpool.hpp>
#include <shark/protocols/nexp.hpp>
#include <shark/protocols/inv.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/utils/globals.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace softmax
        {
            inline void call(u64 s1, u64 s2, const shark::span<u32> &in, shark::span<u64> &out)
            {
                const u64 size = in.size();
                
                shark::span<u32> in_max(s1), delta(size), sum(s1), sum_inv_expanded(size);
                maxpool::max(s1, s2, in, in_max);

                #pragma omp parallel for collapse(2)
                for (u64 i = 0; i < s1; i++)
                {
                    for (u64 j = 0; j < s2; j++)
                    {
                        delta[i * s2 + j] = in[i * s2 + j] - in_max[i];
                    }
                }

                auto exp_x = nexp::call(delta);

                #pragma omp parallel for
                for (u64 i = 0; i < s1; i++)
                {
                    sum[i] = 0;
                    for (u64 j = 0; j < s2; j++)
                    {
                        sum[i] += exp_x[i * s2 + j];
                    }
                }

                auto sum_inv = inv::call(sum);

                #pragma omp parallel for collapse(2)
                for (u64 i = 0; i < s1; ++i)
                {
                    for (u64 j = 0; j < s2; ++j)
                    {
                        sum_inv_expanded[i * s2 + j] = sum_inv[i];
                    }
                }

                auto out_tmp_32 = mul::call(exp_x, sum_inv_expanded);
                // upcase: operation fusion
                shark:span<u64> out_tmp_64(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    out_tmp_64[i] = u64(u64(out_tmp_32[i]) << 32);
                }
                out = truncate::call(out_tmp_64, 32 + FLOAT_PRECISION_32 + FLOAT_PRECISION_32 - FLOAT_PRECISION_64);
            }

            inline shark::span<u64> call(u64 s1, u64 s2, const shark::span<u32> &in)
            {
                always_assert(in.size() == s1 * s2);
                shark::span<u64> out(in.size());
                call(s1, s2, in, out);
                return out;
            }
        }
    }
}