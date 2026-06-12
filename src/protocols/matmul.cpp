#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/eigen.hpp>

using namespace shark::matrix;

namespace shark {
    namespace protocols {
        namespace matmul {

            void gen(u64 a, u64 b, u64 c, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                always_assert(r_X.size() == a * b);
                always_assert(r_Y.size() == b * c);
                always_assert(r_Z.size() == a * c);

                randomize(r_Z);
                auto mat_r_X = getMat(a, b, r_X);
                auto mat_r_Y = getMat(b, c, r_Y);
                auto mat_r_Z = getMat(a, c, r_Z);

                shark::span<u64> r_C(a * c);
                auto mat_r_C = getMat(a, c, r_C);
                // r_C = r_X @ r_Y + r_Z
                // shark::utils::matmuladd(a, b, c, r_X, r_Y, r_Z, r_C);
                mat_r_C = mat_r_X * mat_r_Y + mat_r_Z;

                send_authenticated_ashare(r_X);
                send_authenticated_ashare(r_Y);
                send_authenticated_ashare(r_C);
            }

            void eval(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                always_assert(X.size() == a * b);
                always_assert(Y.size() == b * c);
                always_assert(Z.size() == a * c);

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(a * b);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(b * c);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(a * c);
                shark::utils::stop_timer("key_read");

                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y 
                // shark::span<u128> Z_share(a * c);
                // shark::span<u128> Z_tag(a * c);

                // #pragma omp parallel for collapse(2)
                // for (u64 i = 0; i < a; ++i)
                // {
                //     for (u64 j = 0; j < c; ++j)
                //     {
                //         int index = i * c + j;
                //         u128 zs = r_Z[index];
                //         u128 zt = r_Z_tag[index];

                //         for (u64 k = 0; k < b; ++k)
                //         {
                //             int index_1 = i * b + k;
                //             auto x  = X[index_1];
                //             auto tx = x * u128(party) - r_X[index_1];
                //             auto tt = x * ring_key - r_X_tag[index_1];

                //             int index_2 = k * c + j;
                //             auto y  = Y[index_2];
                //             auto ry = r_Y[index_2];
                //             auto rt = r_Y_tag[index_2];

                //             zs += tx * y - x * ry;
                //             zt += tt * y - x * rt;
                //         }
                        
                //         Z_share[index] = zs;
                //         Z_tag[index]   = zt;
                //     }
                // }

                // Z = authenticated_reconstruct(Z_share, Z_tag);


                auto mat_X = getMat(a, b, X).cast<u128>();
                auto mat_Y = getMat(b, c, Y).cast<u128>();

                shark::span<u128> Z_share(a * c);
                shark::span<u128> Z_tag(a * c);
                auto mat_Z_share = getMat(a, c, Z_share);
                auto mat_Z_tag = getMat(a, c, Z_tag);

                auto mat_r_X = getMat(a, b, r_X);
                auto mat_r_X_tag = getMat(a, b, r_X_tag);
                auto mat_r_Y = getMat(b, c, r_Y);
                auto mat_r_Y_tag = getMat(b, c, r_Y_tag);
                auto mat_r_Z = getMat(a, c, r_Z);
                auto mat_r_Z_tag = getMat(a, c, r_Z_tag);

                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y    
                // Solution one
                // auto tmp1_share = mat_X * u128(party) - mat_r_X;
                // Eigen::Matrix<u128, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp2_share = tmp1_share * mat_Y;
                // mat_Z_share = mat_r_Z + tmp2_share;
                auto tmp_share_1 = ((mat_X * u128(party) - mat_r_X).eval() * mat_Y).eval();
                auto tmp_share_2 = mat_X * mat_r_Y;
                mat_Z_share = mat_r_Z + tmp_share_1 - tmp_share_2;

                // auto tmp1_tag = mat_X * ring_key - mat_r_X_tag;
                // Eigen::Matrix<u128, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp2_tag = tmp1_tag * mat_Y;
                // mat_Z_tag = mat_r_Z_tag + tmp2_tag;
                auto tmp_tag_1 = ((mat_X * ring_key - mat_r_X_tag).eval() * mat_Y).eval();
                auto tmp_tag_2 = mat_X * mat_r_Y_tag;
                mat_Z_tag = mat_r_Z_tag + tmp_tag_1 - tmp_tag_2;

                Z = authenticated_reconstruct(Z_share, Z_tag);
            }

            void call(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    gen(a, b, c, X, Y, Z);
                }
                else
                {
                    eval(a, b, c, X, Y, Z);
                }
            }

            shark::span<u64> call(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);
                call(a, b, c, X, Y, Z);
                return Z;
            }

            shark::span<u64> emul(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);
                auto X_mat = getMat(a, b, X);
                auto Y_mat = getMat(b, c, Y);
                auto Z_mat = getMat(a, c, Z);
                Z_mat = X_mat * Y_mat;
                return Z;
            }
        }
    }
}