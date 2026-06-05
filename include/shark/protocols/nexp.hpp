#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/protocols/lut.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/lookuptable.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace nexp
        {
            const u32 ln2 = u32(std::make_signed_t<u32>(floor(log(2) * (1 << FLOAT_PRECISION_32))));  // ln2
            const u32 ln2ni = u32(std::make_signed_t<u32>(floor((-1.0 / log(2)) * (1 << FLOAT_PRECISION_32))));  // -1/ln2
            const u32 v1353 = u32(std::make_signed_t<u32>(floor(1.353 * (1 << FLOAT_PRECISION_32))));  // 1.353
            const u32 v03585 = u32(std::make_signed_t<u32>(floor(0.3585 * (1 << FLOAT_PRECISION_32))));  // 0.3585
            const u32 v0344 = u32(std::make_signed_t<u32>(floor(0.344 * (1 << FLOAT_PRECISION_32))));  // 0.344

            inline void gen(const shark::span<u32> &r_in, shark::span<u32> &r_out)
            {
                u64 size = r_in.size();

                shark::span<u32> r_z(size), r_scale_minus_z(size);
                shark::span<u32> r_p(size), r_neg_exp2_z(size), r_exp_p(size);

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_z[i] = ln2ni * r_in[i];
                }
                r_z = truncate::call(r_z, FLOAT_PRECISION_32 << 1);  // r_z is the integer part

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_p[i] = ln2 * r_z[i] + r_in[i];  // r_p = r_p + 1.353
                    r_scale_minus_z[i] = u32(0) - r_z[i];
                }

                auto clip_z_tmp = drelu::call(r_z);
                auto r_clip_z = select::call(clip_z_tmp, r_scale_minus_z);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_clip_z[i] += r_z[i];
                }
                lut::call(r_clip_z, r_neg_exp2_z, lut_exp_2, bin_exp_2);

                r_exp_p = mul::call(r_p, r_p);
                r_exp_p = truncate::call(r_exp_p, FLOAT_PRECISION_32);

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_exp_p[i] = r_exp_p[i] * v03585;
                }
                r_exp_p = truncate::call(r_exp_p, FLOAT_PRECISION_32);
                
                r_out = mul::call(r_neg_exp2_z, r_exp_p);
                r_out = truncate::call(r_out, FLOAT_PRECISION_32);
            }

            inline void eval(const shark::span<u32> &in, shark::span<u32> &out)
            {
                const u64 size = in.size();

                shark::span<u32> z(size), z_minus_scale(size), scale_minus_z(size); // these are integers
                shark::span<u32> p(size), neg_exp2_z(size), exp_p(size); // the precisions of these are equal to FLOAT_PRECISION_32

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    z[i] = ln2ni * in[i];
                }
                z = truncate::call(z, FLOAT_PRECISION_32 << 1);  // z is the integer part

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    p[i] = ln2 * z[i] + in[i] + v1353;  // p = p + 1.353
                    z_minus_scale[i] = z[i] - u32(FLOAT_PRECISION_32 + 1);
                    scale_minus_z[i] = u32(FLOAT_PRECISION_32 + 1) - z[i];
                }

                auto clip_z_tmp = drelu::call(z_minus_scale);
                auto clip_z = select::call(clip_z_tmp, scale_minus_z);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    clip_z[i] += z[i];  // min(z,f+1)
                }
                lut::call(clip_z, neg_exp2_z, lut_exp_2, bin_exp_2);

                exp_p = mul::call(p, p);
                exp_p = truncate::call(exp_p, FLOAT_PRECISION_32);

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    exp_p[i] = exp_p[i] * v03585;
                }
                exp_p = truncate::call(exp_p, FLOAT_PRECISION_32);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    exp_p[i] += v0344;
                }
                
                out = mul::call(neg_exp2_z, exp_p);
                out = truncate::call(out, FLOAT_PRECISION_32);
            }

            inline void call(const shark::span<u32> &in, shark::span<u32> &out)
            {
                if (party == DEALER)
                {
                    gen(in, out);
                }
                else
                {
                    eval(in, out);
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