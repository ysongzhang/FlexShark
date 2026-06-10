#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/rsqrt.hpp>
#include <shark/protocols/add.hpp>
#include <shark/utils/globals.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace layernorm
        {
            inline u64 double2fix(double x) {
                return static_cast<u64>(static_cast<std::make_signed_t<u64>>(x * (1 << FLOAT_PRECISION_64)));  // Note: must truncate the decimal part using static_cast<int_t>
            }

            inline void call(u64 n_token, u64 n_embd, const shark::span<u64> &x, const shark::span<u64> &gamma, const shark::span<u64> &beta, shark::span<u64> &res)
            {
                // LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
                u64 factor = double2fix(1.0 / n_embd);

                // 1. Mean = Truncate(Sum * factor)
                shark::span<u64> x_mean(n_token);
                #pragma omp parallel for
                for(int i = 0; i < n_token; ++i) {
                    x_mean[i] = 0;
                    u64* x_row = x.data() + i * n_embd;
                    for(int j = 0; j < n_embd; ++j) {
                        x_mean[i] += x_row[j];
                    }
                    x_mean[i] = x_mean[i] * factor;
                }
                x_mean = truncate::call(x_mean, FLOAT_PRECISION_64);
                
                // 2. Center: x - Mean
                shark::span<u64> x_centered(n_token * n_embd);
                #pragma omp parallel for collapse(2)
                for(int i = 0; i < n_token; ++i) {
                    for(int j = 0; j < n_embd; ++j) {
                        int idx = i * n_embd + j;
                        x_centered[idx] = x[idx] - x_mean[i]; 
                    }
                }
                
                // 3. Square: (x - Mean)^2
                auto x_sq = mul::call(x_centered, x_centered);
                x_sq = truncate::call(x_sq, FLOAT_PRECISION_64);
                
                // 4. Variance = Truncate(SumSq * factor)
                shark::span<u64> variance(n_token);
                #pragma omp parallel for
                for(int i = 0; i < n_token; ++i) {
                    variance[i] = 0;
                    u64* x_sq_row = x_sq.data() + i * n_embd;
                    for(int j = 0; j < n_embd; ++j) {
                        variance[i] += x_sq_row[j];
                    }
                    variance[i] = variance[i] * factor;
                }
                // downcast
                auto variance_32 = truncate::call_64_32(variance, FLOAT_PRECISION_64);

                // 5. InvSqrt: 1 / sqrt(Variance + eps)
                shark::span<u64> inv_std(n_token);
                inv_std = rsqrt::call(variance_32);
                
                // 6. Normalize: (x - Mean) * inv_std
                shark::span<u64> inv_std_broadcast(n_token * n_embd);
                #pragma omp parallel for collapse(2)
                for(int i = 0; i < n_token; ++i) {
                    for(int j = 0; j < n_embd; ++j) {
                        inv_std_broadcast[i * n_embd + j] = inv_std[i];
                    }
                }
                
                res = mul::call(x_centered, inv_std_broadcast);
                res = truncate::call(res, FLOAT_PRECISION_64);
                
                // 7. Affine: * gamma + beta
                shark::span<u64> gamma_broadcast(n_token * n_embd);
                shark::span<u64> beta_broadcast(n_token * n_embd);
                #pragma omp parallel for collapse(2)
                for(int i = 0; i < n_token; ++i) {
                    for(int j = 0; j < n_embd; ++j) {
                        int idx = i * n_embd + j;
                        gamma_broadcast[idx] = gamma[j]; 
                        beta_broadcast[idx] = beta[j];
                    }
                }

                res = mul::call(res, gamma_broadcast);
                res = add::call(res, beta_broadcast);
                res = truncate::call(res, FLOAT_PRECISION_64);
            }

            inline shark::span<u64> call(u64 n_token, u64 n_embd, const shark::span<u64> &in, const shark::span<u64> &gamma, const shark::span<u64> &beta)
            {
                always_assert(in.size() == n_token * n_embd);
                shark::span<u64> out(in.size());
                call(n_token, n_embd, in, gamma, beta, out);
                return out;
            }
        }
    }
}