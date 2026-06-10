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
#include <shark/protocols/layernorm.hpp>

#include <shark/utils/globals.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/assert.hpp>

using namespace shark;
using namespace shark::protocols;

struct BertModel
{
    // config
    u64 n_layers = 12;
    u64 n_heads = 12;
    u64 n_embd = 768;
    u64 n_interm = 3072;
    int logdivisor = 3; // log2(sqrt(n_embd / n_heads))

    // Note: Bias must be keeping 2f precision
    // weights
    std::vector<span<u64>> c_attn_w;
    std::vector<span<u64>> c_attn_b;

    std::vector<span<u64>> c_proj_w;
    std::vector<span<u64>> c_proj_b;

    std::vector<span<u64>> ffn_up_w;
    std::vector<span<u64>> ffn_up_b;

    std::vector<span<u64>> ffn_down_w;
    std::vector<span<u64>> ffn_down_b;

    // LayerNorm weights
    std::vector<span<u64>> ln1_w;
    std::vector<span<u64>> ln1_b;
    std::vector<span<u64>> ln2_w;
    std::vector<span<u64>> ln2_b;

    // Classifier Head weights
    // Classifier (Linear)
    span<u64> classifier_w;
    span<u64> classifier_b;
    u64 num_labels = 2;

    bool accuracy_test = false;

    BertModel(u64 layers=12, u64 heads=12, u64 embd=768, u64 interm=3072, u64 labels=2, bool acc_test=false) 
        : n_layers(layers), n_heads(heads), n_embd(embd), n_interm(interm), num_labels(labels), accuracy_test(acc_test),
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
        
        if (acc_test)
        {
            // Init classifier head
            classifier_w = span<u64>(n_embd * num_labels);    
            classifier_b = span<u64>(num_labels);
        }
    }
};

u64 double2fix(double x) {
    return static_cast<u64>(static_cast<std::make_signed_t<u64>>(x * (1 << FLOAT_PRECISION_64)));  // Note: must truncate the decimal part using static_cast<int_t>
}

span<u16> bert_gelu(span<u16> &x)
{
    return gelu::call(x);
}

span<u64> bert_softmax(u64 a, u64 b, span<u32> &x)
{
    // Use library call
    return softmax::call(a, b, x);
}

span<u64> linear(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    auto res = matmul::call(a, b, c, x, w);
    res = add::call(res, bias);
    res = truncate::call(res, FLOAT_PRECISION_64);
    return res;
}

span<u32> linearDowncast32(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    auto res_tmp = matmul::call(a, b, c, x, w);
    res_tmp = add::call(res_tmp, bias);
    auto res = truncate::call_64_32(res_tmp, FLOAT_PRECISION_64);
    return res;
}

span<u16> linearDowncast16(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
{
    auto res_tmp = matmul::call(a, b, c, x, w);
    res_tmp = add::call(res_tmp, bias);
    auto res = truncate::call_64_16(res_tmp, FLOAT_PRECISION_64);
    return res;
}

span<u64> ffn(span<u64> &x, int layer, BertModel &model)
{
    u64 n_token = x.size() / model.n_embd;

    auto res_tmp = linearDowncast16(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);

    utils::start_timer("nonlinear");
    res_tmp = bert_gelu(res_tmp);
    utils::stop_timer("nonlinear");

    auto res = up::call_16_64(res_tmp, FLOAT_PRECISION_16, FLOAT_PRECISION_64);

    res = linear(n_token, model.n_interm, model.n_embd, res, model.ffn_down_w[layer], model.ffn_down_b[layer]);

    return res;
}

span<u64> bert_layernorm(span<u64> &x, u64 n_token, u64 n_embd, const span<u64> &gamma, const span<u64> &beta)
{
    return layernorm::call(n_token, n_embd, x, gamma, beta);
}

template <typename T>
span<T> view(span<T> &x, int n_token, int n_peices, int idx)
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
span<T> transpose(u64 a, u64 b, span<T> &x)
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
void concat(span<T> &x, span<T> &y, int n_token, int idx)
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

span<u64> mha(span<u64> &x, int layer, BertModel &model, const span<u32> &mask = span<u32>())
{
    u64 n_token = x.size() / model.n_embd;
    auto c = linear(n_token, model.n_embd, model.n_embd * 3, x, model.c_attn_w[layer], model.c_attn_b[layer]);

    auto q = view(c, n_token, 3, 0);
    auto k = view(c, n_token, 3, 1);
    auto v = view(c, n_token, 3, 2);

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

    // Apply Mask if present (Additive Masking)
    // Mask shape: [n_token] (0 for valid, -1000 for padding)
    // Broadcast to [n_heads, n_token, n_token]
    // The mask is applied to the key dimension (j), so each query position i attends to key positions j according to the padding mask
    if (mask.data() != nullptr && mask.size() > 0) {
        // Assume mask is 1D [n_token] and masks keys (columns)
        // The mask values should be 0 for valid tokens and a large negative for padding
        // This ensures padding tokens don't contribute to attention

        // #if (DEBUG)
        // {
        //     // DEBUG: Print mask information
        //     std::cout << "[DEBUG MHA] Applying attention mask, size=" << mask.size() << std::endl;
        //     std::cout << "[DEBUG MHA] Mask values (first 10): ";
        //     for (int j = 0; j < std::min(10, (int)mask.size()); ++j) {
        //         std::cout << fix2double(mask[j]) << " ";
        //     }
        //     std::cout << std::endl;

        //     // Count valid vs padding tokens
        //     int valid_count = 0, padding_count = 0;
        //     for (size_t j = 0; j < mask.size(); ++j) {
        //         if (mask[j] == 0) valid_count++;
        //         else padding_count++;
        //     }
        //     std::cout << "[DEBUG MHA] Valid tokens: " << valid_count
        //             << ", Padding tokens: " << padding_count << std::endl;
        //     }
        // #endif

        #pragma omp parallel for collapse(3)
        for(int h=0; h<model.n_heads; ++h) {
            for(int i=0; i<n_token; ++i) {
                for(int j=0; j<n_token; ++j) {
                    u64 idx = h * n_token * n_token + i * n_token + j;
                    // Add mask to score. For padding positions (mask[j] < 0), this pushes score to -inf
                    all_scores[idx] = all_scores[idx] + mask[j];
                }
            }
        }
    } else {
        #if (DEBUG)
        {
            std::cout << "[DEBUG MHA] No attention mask applied (mask.data()="
                  << mask.data() << ", mask.size()=" << mask.size() << ")" << std::endl;
        }
        #endif
    }

    // 2. Batched Softmax
    utils::start_timer("nonlinear");
    auto all_probs = bert_softmax(model.n_heads * n_token, n_token, all_scores);
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

    c = linear(n_token, model.n_embd, model.n_embd, c, model.c_proj_w[layer], model.c_proj_b[layer]);

    return c;
}

span<u64> layer(span<u64> &x, int layer, BertModel &model, const span<u32> &mask = span<u32>())
{
    u64 n_token = x.size() / model.n_embd;
    // Post-LayerNorm Architecture (Standard BERT)
    // 1. Attention Sublayer
    if (party == SERVER) std::cout << "11111" << std::endl;
    auto attn_out = mha(x, layer, model, mask);
    if (party == SERVER) std::cout << "22222" << std::endl;
    auto res = add::call(x, attn_out);
    auto ln1 = bert_layernorm(res, n_token, model.n_embd, model.ln1_w[layer], model.ln1_b[layer]);
    if (party == SERVER) std::cout << "33333" << std::endl;

    // 2. FFN Sublayer
    auto ffn_out = ffn(ln1, layer, model);
    if (party == SERVER) std::cout << "44444" << std::endl;
    res = add::call(ln1, ffn_out);
    auto ln2 = bert_layernorm(res, n_token, model.n_embd, model.ln2_w[layer], model.ln2_b[layer]);
    if (party == SERVER) std::cout << "55555" << std::endl;

    return ln2;
}

span<u64> bert_classifier(span<u64> &hidden_states, BertModel &model)
{
    // Extract [CLS] token (first token)
    // hidden_states: [n_token, n_embd]
    // CLS: [1, n_embd]
    span<u64> cls_token(model.n_embd);
    
    // Copy first token
    std::memcpy(cls_token.data(), hidden_states.data(), model.n_embd * sizeof(u64));
    
    // Classifier: Linear [1, n_embd] * [n_embd, num_labels] -> [1, num_labels]
    auto logits = linear(1, model.n_embd, model.num_labels, cls_token, model.classifier_w, model.classifier_b);
    
    return logits;
}

span<u64> inference(span<u64> &x, BertModel &model, const span<u32> &mask = span<u32>())
{
    span<u64> y = x;
    for (int i = 0; i < model.n_layers; i++)
    {
        // if (party == SERVER) std::cout << "[INFO] Processing Layer " << (i + 1) << "/" << model.n_layers << "..." << std::endl;

        y = layer(y, i, model, mask);
    }

    if (model.accuracy_test)
    {
        // Run Classifier
        // if (party == SERVER) std::cout << "[INFO] Processing Classifier Head..." << std::endl;
        
        auto logits = bert_classifier(y, model);

        return logits;
    }
    else
    {
        return y;
    }
}

void share_model_weights(BertModel &model)
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
    
    if (model.accuracy_test)
    {
        // Share Classifier Head
        input::call(model.classifier_w, SERVER);
        input::call(model.classifier_b, SERVER);
    }
}
