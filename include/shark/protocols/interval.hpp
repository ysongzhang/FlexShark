#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/b2a.hpp>
#include <shark/utils/globals.hpp>

namespace shark
{
    namespace protocols
    {
        namespace interval
        {
            inline void call(const shark::span<u32> &in, shark::span<u32> &out)
            {
                const u64 size = in.size();
                const u64 nexpb_size = u64(log(pow(2, 2 * FLOAT_PRECISION_32)) / log(SCALE_BASE));
                shark::span<u32> delta(size * nexpb_size); // x - b ^ i for i from 1 to max size

                if (party != DEALER)
                {
                    #pragma omp parallel for
                    for(u64 i = 0; i < size; i++)
                    {
                        for (u64 j = 0; j < nexpb_size; j++)
                        {
                            delta[i * nexpb_size + j] = in[i] - u32(pow(SCALE_BASE, j + 1));
                        }
                    }
                } 
                else
                {
                    #pragma omp parallel for
                    for(u64 i = 0; i < size; i++)
                    {
                        for (u64 j = 0; j < nexpb_size; j++)
                        {
                            delta[i * nexpb_size + j] = in[i];
                        }
                    }
                }

                auto d =  drelu::call(delta);
                auto d_arith = b2a::call_32(d);

                // calculate k+1
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    if (party != DEALER)
                    {
                        out[i] = 1;
                    }
                    else
                    {
                        out[i] = 0;
                    }
                    
                    for (u64 j = 0; j < nexpb_size; j++)
                    {
                        out[i] += d_arith[i * nexpb_size + j];
                    }
                }
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