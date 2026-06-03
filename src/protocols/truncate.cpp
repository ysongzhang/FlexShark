#include <shark/protocols/truncate.hpp>
#include <shark/protocols/common.hpp>
#include <shark/types/u128.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    namespace protocols
    {
        namespace truncate
        {
            // Generates (r_trunc, r_trunc_tag, r_in_msb, r_in_msb_tag) and (r_out, r_out_tag)
            void gen(const shark::span<u64> &r_in, shark::span<u64> &r_out, int f)
            {
                const u64 size = r_in.size();

                // Compute r_in_trunc and r_in_msb
                shark::span<u64> r_trunc(size);
                shark::span<u64> r_in_msb(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i) {
                    u64 val = r_in[i];
                    r_in_msb[i] = (val >> 63) & u64(1);
                    r_trunc[i] = val >> f;
                }

                randomize(r_out);
                send_authenticated_ashare(r_out);
                send_authenticated_ashare(r_trunc);
                send_authenticated_ashare(r_in_msb);
            }

            void gen(const shark::span<u32> &r_in, shark::span<u32> &r_out, int f)
            {
                const u64 size = r_in.size();

                // Compute r_in_trunc and r_in_msb
                shark::span<u32> r_trunc(size);
                shark::span<u32> r_in_msb(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i) {
                    u32 val = r_in[i];
                    r_in_msb[i] = (val >> 31) & u32(1);
                    r_trunc[i] = val >> f;
                }

                randomize(r_out);
                send_authenticated_ashare(r_out);
                send_authenticated_ashare(r_trunc);
                send_authenticated_ashare(r_in_msb);
            }

            void gen(const shark::span<u64> &r_in, shark::span<u32> &r_out, int f)
            {
                const u64 size = r_in.size();
                f = f + FLOAT_PRECISION_64 - FLOAT_PRECISION_32;

                // Compute r_in_trunc and r_in_msb
                shark::span<u64> r_trunc(size);
                shark::span<u64> r_in_msb(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i) {
                    u64 val = r_in[i];
                    r_in_msb[i] = (val >> 63) & u64(1);
                    r_trunc[i] = val >> f;
                }

                randomize(r_out);
                send_authenticated_ashare(r_out);
                send_authenticated_ashare(r_trunc);
                send_authenticated_ashare(r_in_msb);
            }

            void gen(const shark::span<u64> &r_in, shark::span<u16> &r_out, int f)
            {
                const u64 size = r_in.size();
                f = f + FLOAT_PRECISION_64 - FLOAT_PRECISION_16;

                // Compute r_in_trunc and r_in_msb
                shark::span<u64> r_trunc(size);
                shark::span<u64> r_in_msb(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i) {
                    u64 val = r_in[i];
                    r_in_msb[i] = (val >> 63) & u64(1);
                    r_trunc[i] = val >> f;
                }

                randomize(r_out);
                send_authenticated_ashare(r_out);
                send_authenticated_ashare(r_trunc);
                send_authenticated_ashare(r_in_msb);
            }

            void eval(const shark::span<u64> &in, shark::span<u64> &out, int f)
            {
                const int bw = 64;
                const u64 size = in.size();

                // ARS(in) = LRS(in + 2^{k-2}) - 2^{k-f-2}
                shark::span<u64> m_in(size);
                shark::span<u64> m_in_msb(size);
                shark::span<u64> m_in_trunc(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    u64 val = in[i] + (u64(1) << 62);
                    m_in[i] = val;
                    m_in_msb[i] = ((val >> 63) & u64(1));
                    m_in_trunc[i] = val >> f; // logistic right shift
                }

                // in' = in + 2^{k-2},   in' = m_in' - r_in
                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // Read truncation pairs of r_in
                shark::utils::start_timer("key_read");
                auto [r_out_share, r_out_tag] = recv_authenticated_ashare(size);
                auto [r_in_trunc_share, r_in_trunc_tag] = recv_authenticated_ashare(size);
                auto [r_in_msb_share, r_in_msb_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // ARS(in) = LRS(in') - 2^{k-f-2}
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    res_share[i] = u128(u64(1) << (bw - f)) * r_in_msb_share[i] * u128(1 - m_in_msb[i]) - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_share[i];
                    res_tag[i] = u128(u64(1) << (bw - f)) * r_in_msb_tag[i] * u128(1 - m_in_msb[i]) - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_tag[i];

                    // if (m_in_msb[i] == 0)
                    // {
                    //     res_share[i] = u128(u64(1) << (bw - f)) * r_in_msb_share[i] - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_share[i];
                    //     res_tag[i] = u128(u64(1) << (bw - f)) * r_in_msb_tag[i] - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_tag[i];
                    // }
                    // else
                    // {
                    //     res_share[i] = u128(party) * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) - r_in_trunc_share[i] + r_out_share[i];
                    //     res_tag[i] = ring_key * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) - r_in_trunc_tag[i] + r_out_tag[i];
                    // }
                }

                out = authenticated_reconstruct(res_share, res_tag);
            }

            void eval(const shark::span<u32> &in, shark::span<u32> &out, int f)
            {
                const int bw = 32;
                const u64 size = in.size();

                // ARS(in) = LRS(in + 2^{k-2}) - 2^{k-f-2}
                shark::span<u32> m_in(size);
                shark::span<u32> m_in_msb(size);
                shark::span<u32> m_in_trunc(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    u32 val = in[i] + (u32(1) << 30);
                    m_in[i] = val;
                    m_in_msb[i] = ((val >> 31) & u32(1));
                    m_in_trunc[i] = val >> f; // logistic right shift
                }

                // in' = in + 2^{k-2},   in' = m_in' - r_in
                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // Read truncation pairs of r_in
                shark::utils::start_timer("key_read");
                auto [r_out_share, r_out_tag] = recv_authenticated_ashare(size);
                auto [r_in_trunc_share, r_in_trunc_tag] = recv_authenticated_ashare(size);
                auto [r_in_msb_share, r_in_msb_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // ARS(in) = LRS(in') - 2^{k-f-2}
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    res_share[i] = u128(u32(1) << (bw - f)) * r_in_msb_share[i] * u128(1 - m_in_msb[i]) - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) + r_out_share[i];
                    res_tag[i] = u128(u32(1) << (bw - f)) * r_in_msb_tag[i] * u128(1 - m_in_msb[i]) - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) + r_out_tag[i];

                    // if (m_in_msb[i] == 0)
                    // {
                    //     res_share[i] = u128(u32(1) << (bw - f)) * r_in_msb_share[i] - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) + r_out_share[i];
                    //     res_tag[i] = u128(u32(1) << (bw - f)) * r_in_msb_tag[i] - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) + r_out_tag[i];
                    // }
                    // else
                    // {
                    //     res_share[i] = u128(party) * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) - r_in_trunc_share[i] + r_out_share[i];
                    //     res_tag[i] = ring_key * (u128(m_in_trunc[i]) - u128(u32(1) << (bw - f - 2))) - r_in_trunc_tag[i] + r_out_tag[i];
                    // }
                }

                out = authenticated_reconstruct_32(res_share, res_tag);
            }

            void eval(const shark::span<u64> &in, shark::span<u32> &out, int f)
            {
                const int bw = 64;
                const u64 size = in.size();
                f = f + FLOAT_PRECISION_64 - FLOAT_PRECISION_32;

                // ARS(in) = LRS(in + 2^{k-2}) - 2^{k-f-2}
                shark::span<u64> m_in(size);
                shark::span<u64> m_in_msb(size);
                shark::span<u64> m_in_trunc(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    u64 val = in[i] + (u64(1) << 62);
                    m_in[i] = val;
                    m_in_msb[i] = ((val >> 63) & u64(1));
                    m_in_trunc[i] = val >> f; // logistic right shift
                }

                // in' = in + 2^{k-2},   in' = m_in' - r_in
                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // Read truncation pairs of r_in
                shark::utils::start_timer("key_read");
                auto [r_out_share, r_out_tag] = recv_authenticated_ashare(size);
                auto [r_in_trunc_share, r_in_trunc_tag] = recv_authenticated_ashare(size);
                auto [r_in_msb_share, r_in_msb_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // ARS(in) = LRS(in') - 2^{k-f-2}
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    res_share[i] = u128(u64(1) << (bw - f)) * r_in_msb_share[i] * u128(1 - m_in_msb[i]) - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_share[i];
                    res_tag[i] = u128(u64(1) << (bw - f)) * r_in_msb_tag[i] * u128(1 - m_in_msb[i]) - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_tag[i];
                }

                out = authenticated_reconstruct_32(res_share, res_tag);
            }

            void eval(const shark::span<u64> &in, shark::span<u16> &out, int f)
            {
                const int bw = 64;
                const u64 size = in.size();
                f = f + FLOAT_PRECISION_64 - FLOAT_PRECISION_16;

                // ARS(in) = LRS(in + 2^{k-2}) - 2^{k-f-2}
                shark::span<u64> m_in(size);
                shark::span<u64> m_in_msb(size);
                shark::span<u64> m_in_trunc(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    u64 val = in[i] + (u64(1) << 62);
                    m_in[i] = val;
                    m_in_msb[i] = ((val >> 63) & u64(1));
                    m_in_trunc[i] = val >> f; // logistic right shift
                }

                // in' = in + 2^{k-2},   in' = m_in' - r_in
                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // Read truncation pairs of r_in
                shark::utils::start_timer("key_read");
                auto [r_out_share, r_out_tag] = recv_authenticated_ashare(size);
                auto [r_in_trunc_share, r_in_trunc_tag] = recv_authenticated_ashare(size);
                auto [r_in_msb_share, r_in_msb_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                // LRS(in') = LRS(m_in') - LRS(r_in) + 2^{k-f} * ((1 - MSB(m_in')) * MSB(r_in)) + E
                // ARS(in) = LRS(in') - 2^{k-f-2}
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++) {
                    res_share[i] = u128(u64(1) << (bw - f)) * r_in_msb_share[i] * u128(1 - m_in_msb[i]) - r_in_trunc_share[i] + u128(party) * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_share[i];
                    res_tag[i] = u128(u64(1) << (bw - f)) * r_in_msb_tag[i] * u128(1 - m_in_msb[i]) - r_in_trunc_tag[i] + ring_key * (u128(m_in_trunc[i]) - u128(u64(1) << (bw - f - 2))) + r_out_tag[i];
                }

                out = authenticated_reconstruct_16(res_share, res_tag);
            }

            void call(const shark::span<u64> &in, shark::span<u64> &out, int f)
            {
                if (party == DEALER)
                {
                    gen(in, out, f);
                }
                else
                {
                    eval(in, out, f);
                }
            }

            void call(const shark::span<u32> &in, shark::span<u32> &out, int f)
            {
                if (party == DEALER)
                {
                    gen(in, out, f);
                }
                else
                {
                    eval(in, out, f);
                }
            }

            void call(const shark::span<u64> &in, shark::span<u32> &out, int f)
            {
                if (party == DEALER)
                {
                    gen(in, out, f);
                }
                else
                {
                    eval(in, out, f);
                }
            }

            void call(const shark::span<u64> &in, shark::span<u16> &out, int f)
            {
                if (party == DEALER)
                {
                    gen(in, out, f);
                }
                else
                {
                    eval(in, out, f);
                }
            }

            shark::span<u64> call(const shark::span<u64> &in, int f)
            {
                shark::span<u64> out(in.size());
                call(in, out, f);
                return out;
            }
            
            shark::span<u32> call(const shark::span<u32> &in, int f)
            {
                shark::span<u32> out(in.size());
                call(in, out, f);
                return out;
            }

            shark::span<u32> call_64_32(const shark::span<u64> &in, int f)
            {
                shark::span<u32> out(in.size());
                call(in, out, f);
                return out;
            }

            shark::span<u16> call_64_16(const shark::span<u64> &in, int f)
            {
                shark::span<u16> out(in.size());
                call(in, out, f);
                return out;
            }
        }
    }
}