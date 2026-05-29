#include <shark/protocols/down.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/types/u128.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace protocols
    {
        namespace down
        {
            void gen(const shark::span<u64> &r_in, shark::span<u32> &r_out, int f, int f_prime)
            {
                auto y = truncate::call(r_in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < r_in.size(); i++)
                {
                    r_out[i] = (u32) y[i];
                }
            }
            
            void gen(const shark::span<u32> &r_in, shark::span<u16> &r_out, int f, int f_prime)
            {
                auto y = truncate::call(r_in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < r_in.size(); i++)
                {
                    r_out[i] = (u16) y[i];
                }
            }

            void gen(const shark::span<u64> &r_in, shark::span<u16> &r_out, int f, int f_prime)
            {
                auto y = truncate::call(r_in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < r_in.size(); i++)
                {
                    r_out[i] = (u16) y[i];
                }
            }

            void eval(const shark::span<u64> &in, shark::span<u32> &out, int f, int f_prime)
            {
                auto y = truncate::call(in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < in.size(); i++)
                {
                    out[i] = (u32) y[i];
                }
            }
            
            void eval(const shark::span<u32> &in, shark::span<u16> &out, int f, int f_prime)
            {
                auto y = truncate::call(in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < in.size(); i++)
                {
                    out[i] = (u16) y[i];
                }
            }

            void eval(const shark::span<u64> &in, shark::span<u16> &out, int f, int f_prime)
            {
                auto y = truncate::call(in, f - f_prime);
                #pragma omp parallel for
                for (u64 i = 0; i < in.size(); i++)
                {
                    out[i] = (u16) y[i];
                }
            }

            void call(const shark::span<u64> &in, shark::span<u32> &out, int f, int f_prime)
            {
                if (party == DEALER)
                {
                    gen(in, out, f, f_prime);
                }
                else
                {
                    eval(in, out, f, f_prime);
                }
            }

            void call(const shark::span<u32> &in, shark::span<u16> &out, int f, int f_prime)
            {
                if (party == DEALER)
                {
                    gen(in, out, f, f_prime);
                }
                else
                {
                    eval(in, out, f, f_prime);
                }
            }

            void call(const shark::span<u64> &in, shark::span<u16> &out, int f, int f_prime)
            {
                if (party == DEALER)
                {
                    gen(in, out, f, f_prime);
                }
                else
                {
                    eval(in, out, f, f_prime);
                }
            }

            shark::span<u32> call_64_32(const shark::span<u64> &in, int f, int f_prime)
            {
                shark::span<u32> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }

            shark::span<u16> call_32_16(const shark::span<u32> &in, int f, int f_prime)
            {
                shark::span<u16> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }

            shark::span<u16> call_64_16(const shark::span<u64> &in, int f, int f_prime)
            {
                shark::span<u16> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }
        }
    }
}