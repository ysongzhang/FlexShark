#include <shark/protocols/matmulpar.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/eigen.hpp>

using namespace shark::matrix;

namespace shark {
    namespace protocols {
        namespace matmulpar {

            void gen(u64 l, u64 a, u64 b, u64 c, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                const u64 X_block = a * b;
                const u64 Y_block = b * c;
                const u64 Z_block = a * c;
                always_assert(r_X.size() == l * X_block);
                always_assert(r_Y.size() == l * Y_block);
                always_assert(r_Z.size() == l * Z_block);

                randomize(r_Z);

                for (u64 i = 0; i < l; ++i)
                {
                    const auto r_X_i = shark::span<u64>(r_X.data() + i * X_block, X_block);
                    const auto r_Y_i = shark::span<u64>(r_Y.data() + i * Y_block, Y_block);
                    auto r_Z_i = shark::span<u64>(r_Z.data() + i * Z_block, Z_block);

                    auto mat_r_X = getMat(a, b, r_X_i);
                    auto mat_r_Y = getMat(b, c, r_Y_i);
                    auto mat_r_Z = getMat(a, c, r_Z_i);

                    shark::span<u64> r_C(Z_block);
                    auto mat_r_C = getMat(a, c, r_C);
                    // r_C = r_X @ r_Y + r_Z
                    // shark::utils::matmuladd(a, b, c, r_X, r_Y, r_Z, r_C);
                    mat_r_C = mat_r_X * mat_r_Y + mat_r_Z;

                    send_authenticated_ashare(r_X_i);
                    send_authenticated_ashare(r_Y_i);
                    send_authenticated_ashare(r_C);
                }
            }

            void eval(u64 l, u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                const u64 X_block = a * b;
                const u64 Y_block = b * c;
                const u64 Z_block = a * c;
                always_assert(X.size() == l * X_block);
                always_assert(Y.size() == l * Y_block);
                always_assert(Z.size() == l * Z_block);

                shark::span<u128> Z_share_all(l * Z_block);
                shark::span<u128> Z_tag_all(l * Z_block);

                for (u64 i = 0; i < l; ++i)
                {
                    shark::utils::start_timer("key_read");
                    auto [r_X, r_X_tag] = recv_authenticated_ashare(X_block);
                    auto [r_Y, r_Y_tag] = recv_authenticated_ashare(Y_block);
                    auto [r_Z, r_Z_tag] = recv_authenticated_ashare(Z_block);
                    shark::utils::stop_timer("key_read");

                    auto X_i = shark::span<u64>(X.data() + i * X_block, X_block);
                    auto Y_i = shark::span<u64>(Y.data() + i * Y_block, Y_block);
                    auto mat_X = getMat(a, b, X_i).cast<u128>().eval();
                    auto mat_Y = getMat(b, c, Y_i).cast<u128>().eval();

                    shark::span<u128> Z_share(Z_share_all.data() + i * Z_block, Z_block);
                    shark::span<u128> Z_tag(Z_tag_all.data() + i * Z_block, Z_block);
                    auto mat_Z_share = getMat(a, c, Z_share);
                    auto mat_Z_tag = getMat(a, c, Z_tag);

                    auto mat_r_X = getMat(a, b, r_X);
                    auto mat_r_X_tag = getMat(a, b, r_X_tag);
                    auto mat_r_Y = getMat(b, c, r_Y);
                    auto mat_r_Y_tag = getMat(b, c, r_Y_tag);
                    auto mat_r_Z = getMat(a, c, r_Z);
                    auto mat_r_Z_tag = getMat(a, c, r_Z_tag);

                    // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y
                    mat_Z_share = mat_r_Z + (mat_X * u128(party) - mat_r_X) * mat_Y;
                    mat_Z_share -= mat_X * mat_r_Y;

                    mat_Z_tag = mat_r_Z_tag + (mat_X * ring_key - mat_r_X_tag) * mat_Y;
                    mat_Z_tag -= mat_X * mat_r_Y_tag;
                }

                Z = authenticated_reconstruct(Z_share_all, Z_tag_all);
            }

            void call(u64 l, u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    gen(l, a, b, c, X, Y, Z);
                }
                else
                {
                    eval(l, a, b, c, X, Y, Z);
                }
            }

            shark::span<u64> call(u64 l, u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(l * a * c);
                call(l, a, b, c, X, Y, Z);
                return Z;
            }
        }
    }
}
