#include <shark/protocols/b2a.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    namespace protocols
    {
        namespace b2a
        {
            void gen(const shark::span<u8> &r_s, shark::span<u64> &res)
            {
                u64 size = r_s.size();
                always_assert(res.size() == size);
                randomize(res);

                shark::span<u64> u(size);

                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i)
                {
                    u[i] = r_s[i];
                }

                send_authenticated_ashare(u);
                send_authenticated_ashare(res);
            }

            void gen(const shark::span<u8> &r_s, shark::span<u32> &res)
            {
                u64 size = r_s.size();
                always_assert(res.size() == size);
                randomize(res);

                shark::span<u32> u(size);

                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i)
                {
                    u[i] = r_s[i];
                }

                send_authenticated_ashare(u);
                send_authenticated_ashare(res);
            }

            void gen(const shark::span<u8> &r_s, shark::span<u16> &res)
            {
                u64 size = r_s.size();
                always_assert(res.size() == size);
                randomize(res);

                shark::span<u16> u(size);

                #pragma omp parallel for
                for (u64 i = 0; i < size; ++i)
                {
                    u[i] = r_s[i];
                }

                send_authenticated_ashare(u);
                send_authenticated_ashare(res);
            }

            void eval(const shark::span<u8> &s, shark::span<u64> &res)
            {
                u64 size = s.size();
                always_assert(res.size() == size);
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);

                shark::utils::start_timer("key_read");
                auto [u, u_tag] = recv_authenticated_ashare(size);
                auto [v, v_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    if (s[i] == 0)
                    {
                        res_share[i] = u[i] + v[i];
                        res_tag[i] = u_tag[i] + v_tag[i];
                    }
                    else
                    {
                        res_share[i] = party - u[i] + v[i];
                        res_tag[i] = ring_key - u_tag[i] + v_tag[i];
                    }
                }

                res = authenticated_reconstruct(res_share, res_tag);
            }

            void eval(const shark::span<u8> &s, shark::span<u32> &res)
            {
                u64 size = s.size();
                always_assert(res.size() == size);
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);

                shark::utils::start_timer("key_read");
                auto [u, u_tag] = recv_authenticated_ashare(size);
                auto [v, v_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    if (s[i] == 0)
                    {
                        res_share[i] = u[i] + v[i];
                        res_tag[i] = u_tag[i] + v_tag[i];
                    }
                    else
                    {
                        res_share[i] = party - u[i] + v[i];
                        res_tag[i] = ring_key - u_tag[i] + v_tag[i];
                    }
                }

                res = authenticated_reconstruct_32(res_share, res_tag);
            }

            void eval(const shark::span<u8> &s, shark::span<u16> &res)
            {
                u64 size = s.size();
                always_assert(res.size() == size);
                shark::span<u128> res_share(size);
                shark::span<u128> res_tag(size);

                shark::utils::start_timer("key_read");
                auto [u, u_tag] = recv_authenticated_ashare(size);
                auto [v, v_tag] = recv_authenticated_ashare(size);
                shark::utils::stop_timer("key_read");

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    if (s[i] == 0)
                    {
                        res_share[i] = u[i] + v[i];
                        res_tag[i] = u_tag[i] + v_tag[i];
                    }
                    else
                    {
                        res_share[i] = party - u[i] + v[i];
                        res_tag[i] = ring_key - u_tag[i] + v_tag[i];
                    }
                }

                res = authenticated_reconstruct_16(res_share, res_tag);
            }

            void call(const shark::span<u8> &s, shark::span<u64> &res)
            {
                if (party == DEALER)
                {
                    gen(s, res);
                }
                else
                {
                    eval(s, res);
                }
            }

            void call(const shark::span<u8> &s, shark::span<u32> &res)
            {
                if (party == DEALER)
                {
                    gen(s, res);
                }
                else
                {
                    eval(s, res);
                }
            }

            void call(const shark::span<u8> &s, shark::span<u16> &res)
            {
                if (party == DEALER)
                {
                    gen(s, res);
                }
                else
                {
                    eval(s, res);
                }
            }

            shark::span<u64> call(const shark::span<u8> &s)
            {
                shark::span<u64> res(s.size());
                call(s, res);
                return res;
            }

            shark::span<u32> call_32(const shark::span<u8> &s)
            {
                shark::span<u32> res(s.size());
                call(s, res);
                return res;
            }

            shark::span<u16> call_16(const shark::span<u8> &s)
            {
                shark::span<u16> res(s.size());
                call(s, res);
                return res;
            }
        }

    }
}
