#include <shark/protocols/lut.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    namespace protocols
    {
        namespace lut
        {
            void gen(const shark::span<u64> &X, shark::span<u64> &Y, const std::vector<u64> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));
                randomize(Y);

                shark::span<u64> Xneg(X.size());
                for (u64 i = 0; i < X.size(); i++)
                {
                    Xneg[i] = -X[i];
                }

                send_dpfring<u64>(Xneg, bin);
                send_authenticated_ashare(Y);
            }

            void gen(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));
                randomize(Y);

                shark::span<u32> Xneg(X.size());
                for (u64 i = 0; i < X.size(); i++)
                {
                    Xneg[i] = -X[i];
                }

                send_dpfring<u32>(Xneg, bin);
                send_authenticated_ashare(Y);
            }

            void gen(const shark::span<u16> &X, shark::span<u16> &Y, const std::vector<u16> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));
                randomize(Y);

                shark::span<u16> Xneg(X.size());
                for (u64 i = 0; i < X.size(); i++)
                {
                    Xneg[i] = -X[i];
                }

                send_dpfring<u16>(Xneg, bin);
                send_authenticated_ashare(Y);
            }

            void eval(const shark::span<u64> &X, shark::span<u64> &Y, const std::vector<u64> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));

                shark::utils::start_timer("key_read");
                auto dpfKeys = recv_dpfring(X.size(), bin);
                auto [R_share, R_tag] = recv_authenticated_ashare(X.size());
                shark::utils::stop_timer("key_read");

                shark::span<u128> Y_share(X.size());
                shark::span<u128> Y_tag(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); i++)
                {
                    auto x = X[i];

                    auto [res_share, res_tag] = crypto::dpfring_evalall_reduce<u64>(party, dpfKeys[i], lut, x);

                    Y_share[i] = res_share + R_share[i];
                    Y_tag[i] = res_tag + R_tag[i];
                }

                Y = authenticated_reconstruct(Y_share, Y_tag);
            }

            void eval(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));

                shark::utils::start_timer("key_read");
                auto dpfKeys = recv_dpfring(X.size(), bin);
                auto [R_share, R_tag] = recv_authenticated_ashare(X.size());
                shark::utils::stop_timer("key_read");

                shark::span<u128> Y_share(X.size());
                shark::span<u128> Y_tag(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); i++)
                {
                    auto x = X[i];

                    auto [res_share, res_tag] = crypto::dpfring_evalall_reduce<u32>(party, dpfKeys[i], lut, x);

                    Y_share[i] = res_share + R_share[i];
                    Y_tag[i] = res_tag + R_tag[i];
                }

                Y = authenticated_reconstruct_32(Y_share, Y_tag);
            }

            void eval(const shark::span<u16> &X, shark::span<u16> &Y, const std::vector<u16> &lut, int bin)
            {
                always_assert(X.size() == Y.size());
                always_assert(lut.size() == (1ull << bin));

                shark::utils::start_timer("key_read");
                auto dpfKeys = recv_dpfring(X.size(), bin);
                auto [R_share, R_tag] = recv_authenticated_ashare(X.size());
                shark::utils::stop_timer("key_read");

                shark::span<u128> Y_share(X.size());
                shark::span<u128> Y_tag(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); i++)
                {
                    auto x = X[i];

                    auto [res_share, res_tag] = crypto::dpfring_evalall_reduce<u16>(party, dpfKeys[i], lut, x);

                    Y_share[i] = res_share + R_share[i];
                    Y_tag[i] = res_tag + R_tag[i];
                }

                Y = authenticated_reconstruct_16(Y_share, Y_tag);
            }

            void call(const shark::span<u64> &X, shark::span<u64> &Y, const std::vector<u64> &lut, int bin)
            {
                if (party == DEALER)
                {
                    gen(X, Y, lut, bin);
                }
                else
                {
                    eval(X, Y, lut, bin);
                }
            }

            void call(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut, int bin)
            {
                if (party == DEALER)
                {
                    gen(X, Y, lut, bin);
                }
                else
                {
                    eval(X, Y, lut, bin);
                }
            }

            void call(const shark::span<u16> &X, shark::span<u16> &Y, const std::vector<u16> &lut, int bin)
            {
                if (party == DEALER)
                {
                    gen(X, Y, lut, bin);
                }
                else
                {
                    eval(X, Y, lut, bin);
                }
            }

            shark::span<u64> call(const shark::span<u64> &X, const std::vector<u64> &lut, int bin)
            {
                shark::span<u64> Y(X.size());
                call(X, Y, lut, bin);
                return Y;
            }

            shark::span<u32> call(const shark::span<u32> &X, const std::vector<u32> &lut, int bin)
            {
                shark::span<u32> Y(X.size());
                call(X, Y, lut, bin);
                return Y;
            }

            shark::span<u16> call(const shark::span<u16> &X, const std::vector<u16> &lut, int bin)
            {
                shark::span<u16> Y(X.size());
                call(X, Y, lut, bin);
                return Y;
            }


            void gen(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin)
            {
                always_assert((X.size() << 1) == Y.size());
                always_assert(lut_1.size() == (1ull << bin));
                always_assert(lut_2.size() == (1ull << bin));

                randomize(Y);

                shark::span<u32> Xneg(X.size());
                for (u64 i = 0; i < X.size(); i++)
                {
                    Xneg[i] = -X[i];
                }

                send_dpfring<u32>(Xneg, bin);
                send_authenticated_ashare(Y);
            }

            void eval(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin)
            {
                u64 size = X.size();
                always_assert((size << 1) == Y.size());
                always_assert(lut_1.size() == (1ull << bin));
                always_assert(lut_2.size() == (1ull << bin));

                shark::utils::start_timer("key_read");
                auto dpfKeys = recv_dpfring(size, bin);
                auto [R_share, R_tag] = recv_authenticated_ashare(Y.size());
                shark::utils::stop_timer("key_read");

                shark::span<u128> Y_share(Y.size());
                shark::span<u128> Y_tag(Y.size());

                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    auto x = X[i];

                    auto [res_share_1_tup, res_share_2_tup] = crypto::dpfring_evalall_reduce(party, dpfKeys[i], lut_1, lut_2, x);

                    auto [res_share_1, res_tag_1] = res_share_1_tup;
                    auto [res_share_2, res_tag_2] = res_share_2_tup;

                    Y_share[i] = res_share_1 + R_share[i];
                    Y_tag[i] = res_tag_1 + R_tag[i];
                    Y_share[i + size] = res_share_2 + R_share[i + size];
                    Y_tag[i + size] = res_tag_2 + R_tag[i + size];
                }

                Y = authenticated_reconstruct_32(Y_share, Y_tag);
            }

            void call(const shark::span<u32> &X, shark::span<u32> &Y, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin)
            {
                if (party == DEALER)
                {
                    gen(X, Y, lut_1, lut_2, bin);
                }
                else
                {
                    eval(X, Y, lut_1, lut_2, bin);
                }
            }

            shark::span<u32> call(const shark::span<u32> &X, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int bin)
            {
                shark::span<u32> Y(X.size() << 1);
                call(X, Y, lut_1, lut_2, bin);
                return Y;
            }
        }

    }
}
