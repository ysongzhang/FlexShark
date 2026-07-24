#pragma once

#include <cstring>
#include <shark/protocols/common.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/matmulpar.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/inv.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/protocols/gelu.hpp>
#include <shark/protocols/rsqrt.hpp>
#include <shark/protocols/up.hpp>
#include <shark/protocols/down.hpp>
#include <shark/protocols/layernorm.hpp>

#include <shark/utils/globals.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/assert.hpp>

using namespace shark;
using namespace shark::protocols;

struct GPT2Model
{
    // config
    u64 n_layers = 12;
    u64 n_heads = 12;
    u64 n_embd = 768;
    u64 n_interm = 3072;
    int logdivisor = 3; // log2(sqrt(n_embd / n_heads))

    // weights
    std::vector<span<u64>> c_attn_w;
    std::vector<span<u64>> c_attn_b;

    std::vector<span<u64>> c_proj_w;
    std::vector<span<u64>> c_proj_b;

    std::vector<span<u64>> ffn_up_w;
    std::vector<span<u64>> ffn_up_b; // Note: ffn_up_b must be keeping 2f precision

    std::vector<span<u64>> ffn_down_w;
    std::vector<span<u64>> ffn_down_b;

    // LayerNorm weights
    std::vector<span<u64>> ln1_w;
    std::vector<span<u64>> ln1_b;
    std::vector<span<u64>> ln2_w;
    std::vector<span<u64>> ln2_b;

    // Final LayerNorm
    span<u64> ln_f_w;
    span<u64> ln_f_b;

    bool accuracy_test = false;

    GPT2Model(u64 layers=12, u64 heads=12, u64 embd=768, u64 interm=3072, bool acc_test=false) 
        : n_layers(layers), n_heads(heads), n_embd(embd), n_interm(interm), accuracy_test(acc_test),
          c_attn_w(n_layers), c_attn_b(n_layers), c_proj_w(n_layers), c_proj_b(n_layers), 
          ffn_up_w(n_layers), ffn_up_b(n_layers), ffn_down_w(n_layers), ffn_down_b(n_layers),
          ln1_w(n_layers), ln1_b(n_layers), ln2_w(n_layers), ln2_b(n_layers)
    {
        for (int i = 0; i < n_layers; ++i)
        {
            c_attn_w[i] = span<u64>(n_embd * 3 * n_embd);
            c_attn_b[i] = span<u64>(n_embd * 3);
            c_proj_w[i] = span<u64>(n_embd * n_embd);
            c_proj_b[i] = span<u64>(n_embd);
            ffn_up_w[i] = span<u64>(n_embd * n_interm);
            ffn_up_b[i] = span<u64>(n_interm);
            ffn_down_w[i] = span<u64>(n_interm * n_embd);
            ffn_down_b[i] = span<u64>(n_embd);
            
            ln1_w[i] = span<u64>(n_embd);
            ln1_b[i] = span<u64>(n_embd);
            ln2_w[i] = span<u64>(n_embd);
            ln2_b[i] = span<u64>(n_embd);
        }

        ln_f_w = span<u64>(n_embd);
        ln_f_b = span<u64>(n_embd);
    }
};

template <typename T>
span<T> gpt2_gelu(span<T> &x)
{
    return gelu::call(x);
}

span<u64> gpt2_masked_softmax(u64 n_rows, u64 n_cols, span<u32> &x)
{
    const u64 size = x.size();
    u64 n_heads = n_rows / n_cols;

    // 1. Causal mask: copy score at position 0 to all masked positions (j > i)
    //    This is MPC-safe (local copy of shares preserves sharing),
    //    avoids large negative constants, and keeps values in valid range
    #pragma omp parallel for collapse(3)
    for(u64 h = 0; h < n_heads; ++h) {
        for(u64 i = 0; i < n_cols; ++i) {
            for(u64 j = 0; j < n_cols; ++j) {
                if (j > i) {
                    u64 row_base = h * n_cols * n_cols + i * n_cols;
                    x[row_base + j] = x[row_base]; // Copy first element's share
                }
            }
        }
    }

    // 2. Secure max using MPC comparison protocol (NOT local share comparison)
    span<u32> x_max(n_rows);
    maxpool::max(n_rows, n_cols, x, x_max);

    // 3. Exp(x - max)
    span<u32> delta(n_rows * n_cols);
    #pragma omp parallel for collapse(2)
    for(u64 i = 0; i < n_rows; ++i) {
        for(u64 j = 0; j < n_cols; ++j) {
            delta[i * n_cols + j] = x[i * n_cols + j] - x_max[i];
        }
    }
    auto exp_x = nexp::call(delta);

    // 4. Sum only valid (causal) positions: j <= i
    span<u32> sum(n_rows);
    #pragma omp parallel for
    for(u64 i = 0; i < n_rows; ++i) {
        u32 row_sum = 0;
        u64 token_idx = i % n_cols;
        u64 row_offset = i * n_cols;

        for(u64 j = 0; j <= token_idx; ++j) {
            row_sum += exp_x[row_offset + j];
        }
        sum[i] = row_sum;
    }

    // 5. Secure inverse
    auto sum_inv = inv::call(sum);

    // 6. Broadcast & secure multiply
    span<u32> sum_inv_expanded(n_rows * n_cols);
    #pragma omp parallel for collapse(2)
    for(u64 i = 0; i < n_rows; ++i) {
        for(u64 j = 0; j < n_cols; ++j) {
            sum_inv_expanded[i * n_cols + j] = sum_inv[i];
        }
    }

    span<u64> res(n_rows * n_cols);
    auto out_tmp_32 = mul::call(exp_x, sum_inv_expanded);
    // upcast: operation fusion
    shark:span<u64> out_tmp_64(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        out_tmp_64[i] = u64(u64(out_tmp_32[i]) << 32);
    }
    res = truncate::call(out_tmp_64, 32 + FLOAT_PRECISION_32 + FLOAT_PRECISION_32 - FLOAT_PRECISION_64);

    // 7. Zero out masked positions (both parties set share to 0 → reconstructed = 0)
    #pragma omp parallel for collapse(3)
    for(u64 h = 0; h < n_heads; ++h) {
        for(u64 i = 0; i < n_cols; ++i) {
            for(u64 j = 0; j < n_cols; ++j) {
                if (j > i) {
                    res[h * n_cols * n_cols + i * n_cols + j] = 0;
                }
            }
        }
    }

    return res;
}

span<u64> gpt2_linear(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    auto res = matmul::call(a, b, c, x, w);
    res = truncate::call(res, FLOAT_PRECISION_64);
    res = add::call(res, bias);
    return res;
}

span<u32> gpt2_linearDowncast32(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    // Method 1: operation fusion
    auto res_tmp = matmul::call(a, b, c, x, w);
    res_tmp = add::call(res_tmp, bias);
    auto res = truncate::call_64_32(res_tmp, FLOAT_PRECISION_64);
    return res;

    // Method 2: do not double the precision of bias, extra one round
    // auto res_tmp = matmul::call(a, b, c, x, w);
    // res_tmp = truncate::call(res_tmp, FLOAT_PRECISION_64);
    // res_tmp = add::call(res_tmp, bias);
    // auto res = down::call_64_32(res_tmp, FLOAT_PRECISION_64, FLOAT_PRECISION_32);
    // return res;
}

span<u16> gpt2_linearDowncast16(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    // Method 1: operation fusion
    auto res_tmp = matmul::call(a, b, c, x, w);
    res_tmp = add::call(res_tmp, bias);
    auto res = truncate::call_64_16(res_tmp, FLOAT_PRECISION_64);
    return res;

    // Method 2: do not double the precision of bias, extra one round
    // auto res_tmp = matmul::call(a, b, c, x, w);
    // res_tmp = truncate::call(res_tmp, FLOAT_PRECISION_64);
    // res_tmp = add::call(res_tmp, bias);
    // auto res = down::call_64_16(res_tmp, FLOAT_PRECISION_64, FLOAT_PRECISION_16);
    // return res;
}

span<u64> gpt2_ffn(span<u64> &x, int layer, GPT2Model &model)
{
    u64 n_token = x.size() / model.n_embd;

    auto res_tmp = gpt2_linearDowncast16(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);
    // auto res_tmp = gpt2_linearDowncast32(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);

    utils::start_timer("nonlinear");
    res_tmp = gpt2_gelu(res_tmp);
    utils::stop_timer("nonlinear");

    // auto res = up::call_32_64(res_tmp, FLOAT_PRECISION_32, FLOAT_PRECISION_64);
    utils::start_timer("up");
    auto res = up::call_16_64(res_tmp, FLOAT_PRECISION_16, FLOAT_PRECISION_64);
    utils::stop_timer("up");

    res = gpt2_linear(n_token, model.n_interm, model.n_embd, res, model.ffn_down_w[layer], model.ffn_down_b[layer]);

    return res;
}

span<u64> gpt2_layernorm(span<u64> &x, u64 n_token, u64 n_embd, const span<u64> &gamma, const span<u64> &beta)
{
    return layernorm::call(n_token, n_embd, x, gamma, beta);
}

template <typename T>
span<T> gpt2_view(span<T> &x, int n_token, int n_peices, int idx)
{
    always_assert(x.size() % (n_token * n_peices) == 0);
    int n_embd = x.size() / n_token;
    span<T> y(x.size() / n_peices);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_token; i++)
    {
        for (int j = 0; j < (n_embd / n_peices); j++)
        {
            y[i * (n_embd / n_peices) + j] = x[i * n_embd + j + idx * (n_embd / n_peices)];
        }
    }

    return y;
}

template <typename T>
span<T> gpt2_transpose(u64 a, u64 b, span<T> &x)
{
    always_assert(x.size() == (a * b));
    span<T> y(x.size());
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < b; j++)
        {
            y[j * a + i] = x[i * b + j];
        }
    }

    return y;
}

template <typename T>
void gpt2_concat(span<T> &x, span<T> &y, int n_token, int idx)
{
    int x_n_embd = x.size() / n_token;
    int y_n_embd = y.size() / n_token;
    int n_peices = x_n_embd / y_n_embd;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_token; i++)
    {
        for (int j = 0; j < y_n_embd; j++)
        {
            x[i * x_n_embd + j + idx * y_n_embd] = y[i * y_n_embd + j];
        }
    }
}

span<u64> gpt2_mha(span<u64> &x, int layer, GPT2Model &model, const span<u32> &mask = span<u32>())
{
    u64 n_token = x.size() / model.n_embd;
    auto c = gpt2_linear(n_token, model.n_embd, model.n_embd * 3, x, model.c_attn_w[layer], model.c_attn_b[layer]);

    auto q = gpt2_view(c, n_token, 3, 0);
    auto k = gpt2_view(c, n_token, 3, 1);
    auto v = gpt2_view(c, n_token, 3, 2);

    // Batched MHA Optimization:
    u64 head_size = model.n_embd / model.n_heads;
    
    // 1. Compute Attention Scores (Q * K^T)
    // Q_i: [n_token, head_size]
    // K_i^T: [head_size, n_token]
    span<u64> batched_q(model.n_heads * n_token * head_size);
    span<u64> batched_kt(model.n_heads * head_size * n_token);
    span<u64> batched_v(model.n_heads * n_token * head_size);

    // Optimized Q and V preparation
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < model.n_heads; i++)
    {
        for(int r = 0; r < n_token; ++r) {
            u64 dst_offset = i * n_token * head_size + r * head_size;
            u64 src_offset = r * model.n_embd + i * head_size;
            std::memcpy(batched_q.data() + dst_offset, q.data() + src_offset, head_size * sizeof(u64));
            std::memcpy(batched_v.data() + dst_offset, v.data() + src_offset, head_size * sizeof(u64));
        }
    }
    
    // K preparation requires transpose, so kept element-wise copy
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < model.n_heads; ++i) {
        for (int c = 0; c < head_size; ++c) {
            u64* batched_kt_pointer = batched_kt.data() + i * head_size * n_token + c * n_token;
            int idx = i * head_size + c;
            for (int r = 0; r < n_token; ++r) {
                batched_kt_pointer[r] = k[r * model.n_embd + idx];
            }
        }
    }
    
    // Batched MatMul: [n_heads, n_token, head_size] * [n_heads, head_size, n_token] -> [n_heads, n_token, n_token]
    auto batched_scores_raw = matmulpar::call(model.n_heads, n_token, head_size, n_token, batched_q, batched_kt);
    
    // Truncate (scale down by f + logdivisor)
    auto all_scores = truncate::call_64_32(batched_scores_raw, FLOAT_PRECISION_64 + model.logdivisor);

    // 2. Batched Softmax
    utils::start_timer("nonlinear");
    auto all_probs = gpt2_masked_softmax(model.n_heads * n_token, n_token, all_scores);
    utils::stop_timer("nonlinear");

    // 3. Batched Prob * V
    span<u64> qks_sm_vs(n_token * model.n_embd);
    
    // Batched MatMul: Probs * V
    auto batched_res = matmulpar::call(model.n_heads, n_token, n_token, head_size, all_probs, batched_v);
    // Concat
    #pragma omp parallel for collapse(2)
    for(int r = 0; r < n_token; ++r) {
        for(int i = 0; i < model.n_heads; ++i) {
            u64 dst_offset = r * model.n_embd + i * head_size;
            u64 src_offset = i * n_token * head_size + r * head_size;
            std::memcpy(qks_sm_vs.data() + dst_offset, batched_res.data() + src_offset, head_size * sizeof(u64));
        }
    }

    c = truncate::call(qks_sm_vs, FLOAT_PRECISION_64);

    c = gpt2_linear(n_token, model.n_embd, model.n_embd, c, model.c_proj_w[layer], model.c_proj_b[layer]);

    return c;
}

span<u64> gpt2_layer(span<u64> &x, int layer, GPT2Model &model)
{
    u64 n_token = x.size() / model.n_embd;
    // Post-LayerNorm Architecture (Standard GPT-2)
    // 1. Attention Sublayer
    utils::start_timer("layernorm");
    auto ln1 = gpt2_layernorm(x, n_token, model.n_embd, model.ln1_w[layer], model.ln1_b[layer]);
    utils::stop_timer("layernorm");
    utils::start_timer("mha");
    auto attn_out = gpt2_mha(ln1, layer, model);
    utils::stop_timer("mha");
    x = add::call(x, attn_out);
    
    // 2. FFN Sublayer
    utils::start_timer("layernorm");
    auto ln2 = gpt2_layernorm(x, n_token, model.n_embd, model.ln2_w[layer], model.ln2_b[layer]);
    utils::stop_timer("layernorm");
    utils::start_timer("ffn");
    auto ffn_out = gpt2_ffn(ln2, layer, model);
    utils::stop_timer("ffn");
    x = add::call(x, ffn_out);

    return x;
}

span<u64> gpt2_inference(span<u64> &x, GPT2Model &model)
{
    span<u64> y = x;
    for (int i = 0; i < model.n_layers; i++)
    {
        // if (party == SERVER) std::cout << "[INFO] Processing Layer " << (i + 1) << "/" << model.n_layers << "..." << std::endl;

        y = gpt2_layer(y, i, model);
    }

    u64 n_token = y.size() / model.n_embd;
    y = gpt2_layernorm(y, n_token, model.n_embd, model.ln_f_w, model.ln_f_b);

    return y;
}

void share_model_weights(GPT2Model &model)
{
    // if (party == SERVER) std::cout << "[INFO] Sharing model weights..." << std::endl;
    
    for (int i = 0; i < model.n_layers; i++)
    {
        input::call(model.c_attn_w[i], SERVER);
        input::call(model.c_attn_b[i], SERVER);
        input::call(model.c_proj_w[i], SERVER);
        input::call(model.c_proj_b[i], SERVER);
        input::call(model.ffn_up_w[i], SERVER);
        input::call(model.ffn_up_b[i], SERVER);
        input::call(model.ffn_down_w[i], SERVER);
        input::call(model.ffn_down_b[i], SERVER);
        
        input::call(model.ln1_w[i], SERVER);
        input::call(model.ln1_b[i], SERVER);
        input::call(model.ln2_w[i], SERVER);
        input::call(model.ln2_b[i], SERVER);
    }
}
