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
        namespace rsqrt
        {
            inline void call(const shark::span<u32> &in, shark::span<u64> &out)
            {
                const u64 size = in.size();
                
                shark::span<u32> k(size);
                interval::call(in, k);
                shark::span<u32> ts(size << 1);
                lut::call(k, ts, lut_exp_b, lut_exsqrt, bin_exp_b);  // t||s in the paper
                shark::span<u32> t(ts.data(), size);
                shark::span<u32> s(ts.data() + size, size);
                auto q = mul::call(in, t);
                q = truncate::call(q, FLOAT_PRECISION_32);
                u64 rb = u32((1.0 / SCALE_BASE) * (1 << FLOAT_PRECISION_32));
                if(party != DEALER)
                {
                    #pragma omp parallel for
                    for(u64 i = 0; i < size; i++)
                    {
                        q[i] -= rb;
                    }
                }
                q = lut::call(q, lut_rsqrt, bin_rsqrt);
                auto out_tmp_32 = mul::call(s, q);
                // upcase: operation fusion
                shark:span<u64> out_tmp_64(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    out_tmp_64[i] = u64(u64(out_tmp_32[i]) << 32);
                }
                out = truncate::call(out_tmp_64, 32 + FLOAT_PRECISION_32 + FLOAT_PRECISION_32 - FLOAT_PRECISION_64);
            }

            inline shark::span<u64> call(const shark::span<u32> &in)
            {
                shark::span<u64> out(in.size());
                call(in, out);
                return out;
            }
        }
    }
}