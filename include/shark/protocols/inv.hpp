#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/lut.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/interval.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/lookuptable.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace inv
        {
            inline void call(const shark::span<u32> &in, shark::span<u32> &out)
            {
                const u64 size = in.size();
                
                shark::span<u32> k(size);
                interval::call(in, k);
                auto t = lut::call(k, lut_exp_b, bin_exp_b);
                auto tmp_q = mul::call(in, t);
                auto q = truncate::call(tmp_q, FLOAT_PRECISION_32);
                if (party != DEALER)
                {
                    u32 rb = u32((1.0 / SCALE_BASE) * (1 << FLOAT_PRECISION_32));
                    #pragma omp parallel for
                    for(u64 i = 0; i < size; i++)
                    {
                        q[i] -= rb;
                    }
                }

                auto w = lut::call(q, lut_inv, bin_inv);
                auto tmp_out = mul::call(t, w);
                out = truncate::call(tmp_out, FLOAT_PRECISION_32);
            }

            inline shark::span<u32> call(const shark::span<u32> &in)
            {
                shark::span<u32> out(in.size());
                call(in, out);
                return out;
            }
        }
    }
}