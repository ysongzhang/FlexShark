#include <shark/protocols/up.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/types/u128.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace protocols
    {
        namespace up
        {
            void gen(const shark::span<u32> &r_in, shark::span<u64> &r_out, int f, int f_prime)
            {
                u64 size = r_in.size();
                shark:span<u64> r_y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_y[i] = u64(u64(r_in[i]) << 32);
                }

                r_out = truncate::call(r_y, 32 + f - f_prime);
            }
            
            void gen(const shark::span<u16> &r_in, shark::span<u64> &r_out, int f, int f_prime)
            {
                u64 size = r_in.size();
                shark:span<u64> r_y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_y[i] = u64(u64(r_in[i]) << 48);
                }

                r_out = truncate::call(r_y, 48 + f - f_prime);
            }

            void gen(const shark::span<u16> &r_in, shark::span<u32> &r_out, int f, int f_prime)
            {
                u64 size = r_in.size();
                shark:span<u32> r_y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    r_y[i] = u32(u32(r_in[i]) << 16);
                }

                r_out = truncate::call(r_y, 16 + f - f_prime);
            }

            void eval(const shark::span<u32> &in, shark::span<u64> &out, int f, int f_prime)
            {
                u64 size = in.size();
                shark:span<u64> y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    y[i] = u64(u64(in[i]) << 32);
                }

                out = truncate::call(y, 32 + f - f_prime);
            }
            
            void eval(const shark::span<u16> &in, shark::span<u64> &out, int f, int f_prime)
            {
                u64 size = in.size();
                shark:span<u64> y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    y[i] = u64(u64(in[i]) << 48);
                }

                out = truncate::call(y, 48 + f - f_prime);
            }

            void eval(const shark::span<u16> &in, shark::span<u32> &out, int f, int f_prime)
            {
                u64 size = in.size();
                shark:span<u32> y(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    y[i] = u32(u32(in[i]) << 16);
                }

                out = truncate::call(y, 16 + f - f_prime);
            }

            void call(const shark::span<u32> &in, shark::span<u64> &out, int f, int f_prime)
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

            void call(const shark::span<u16> &in, shark::span<u64> &out, int f, int f_prime)
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

            void call(const shark::span<u16> &in, shark::span<u32> &out, int f, int f_prime)
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

            shark::span<u64> call_32_64(const shark::span<u32> &in, int f, int f_prime)
            {
                shark::span<u64> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }

             shark::span<u64> call_16_64(const shark::span<u16> &in, int f, int f_prime)
            {
                shark::span<u64> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }

             shark::span<u32> call_16_32(const shark::span<u16> &in, int f, int f_prime)
            {
                shark::span<u32> out(in.size());
                call(in, out, f, f_prime);
                return out;
            }
        }
    }
}