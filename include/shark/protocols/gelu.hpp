#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/protocols/lut.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/lookuptable.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace gelu
        {
            inline void call(const shark::span<u16> &in, shark::span<u16> &out)
            {
                const u64 size = in.size();
                
                auto p = relu::call(in);
                shark::span<u16> a(size), delta(size);
                #pragma omp parallel for
                for(u64 i = 0; i < size; i++)
                {
                    a[i] = 2 * p[i] - in[i];
                    if (party != DEALER)
                    {
                        delta[i] = u16(256) - a[i];
                    }
                    else
                    {
                        delta[i] = u16(0) - a[i];
                    }
                }
                auto i_tmp = drelu::call(delta);
                auto i = select::call(i_tmp, a);
                auto out_tmp = lut::call(i, lut_gelu, bin_gelu);
                #pragma omp parallel for
                for(u64 i = 0; i < size; i++)
                {
                    out[i] = p[i] - out_tmp[i];
                }
            }

            inline shark::span<u16> call(const shark::span<u16> &in)
            {
                shark::span<u16> out(in.size());
                call(in, out);
                return out;
            }
        }
    }
}