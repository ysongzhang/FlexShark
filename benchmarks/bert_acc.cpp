#include "bert_model.hpp"
#include "shark_loader.hpp"

using namespace shark;
using namespace shark::protocols;

void load_model_weights(BertModel &model, SharkLoader &loader)
{

    for (int i = 0; i < model.n_layers; i++)
    {
        // 1. Q, K, V
        auto q = loader.pop();
        auto k = loader.pop();
        auto v = loader.pop();
        
        // Merge Q, K, V into c_attn and transpose
        // PyTorch [Out, In] -> BertModel [In, Out]
        
        int n_embd = model.n_embd;
        
        // Transpose and copy
        auto transpose_merge = [&](const std::vector<u64>& src, span<u64>& dest, int rows, int cols, int dest_width, int col_offset) {
             if (src.size() != rows * cols) {
                 std::cerr << "[Error] transpose_merge size mismatch! Expected " << rows * cols << ", got " << src.size() << std::endl;
                 exit(1);
             }
             #pragma omp parallel for collapse(2)
             for (int r = 0; r < rows; r++) {
                 for (int c = 0; c < cols; c++) {
                     dest[r * dest_width + c + col_offset] = (u64)src[c * rows + r];
                 }
             }
        };

        transpose_merge(q.weight, model.c_attn_w[i], n_embd, n_embd, 3*n_embd, 0);
        transpose_merge(k.weight, model.c_attn_w[i], n_embd, n_embd, 3*n_embd, n_embd);
        transpose_merge(v.weight, model.c_attn_w[i], n_embd, n_embd, 3*n_embd, 2*n_embd);
        
        // Biases: Concatenate
        auto copy_merge = [&](const std::vector<u64>& src, span<u64>& dest, int offset) {
            for (size_t j = 0; j < src.size(); j++) {
                dest[offset + j] = (u64)src[j];
            }
        };
        copy_merge(q.bias, model.c_attn_b[i], 0);
        copy_merge(k.bias, model.c_attn_b[i], n_embd);
        copy_merge(v.bias, model.c_attn_b[i], 2*n_embd);

        // 4. Dense (Attn Output) -> c_proj
        auto p = loader.pop();
        
        auto process_weight = [&](const std::vector<u64>& src, span<u64>& dest, int rows, int cols) {
             if (src.size() != rows * cols) {
                 std::cerr << "[Error] process_weight size mismatch! Expected " << rows * cols << ", got " << src.size() << std::endl;
                 exit(1);
             }
             #pragma omp parallel for collapse(2)
             for (int r = 0; r < rows; r++) {
                 for (int c = 0; c < cols; c++) {
                     dest[r * cols + c] = (u64)src[c * rows + r];
                 }
             }
        };
        process_weight(p.weight, model.c_proj_w[i], n_embd, n_embd);
        fill_span(model.c_proj_b[i], p.bias);

        // 5. LayerNorm1 -> ln1
        auto ln1 = loader.pop();
        fill_span(model.ln1_w[i], ln1.weight);
        fill_span(model.ln1_b[i], ln1.bias);

        // 6. Dense0 (FFN Up) -> ffn_up
        // PyTorch [n_interm, n_embd] -> Dest [n_embd, n_interm]
        auto ffn0 = loader.pop();
        process_weight(ffn0.weight, model.ffn_up_w[i], n_embd, model.n_interm);
        fill_span(model.ffn_up_b[i], ffn0.bias);

        // 7. Dense1 (FFN Down) -> ffn_down
        // PyTorch [n_embd, n_interm] -> Dest [n_interm, n_embd]
        auto ffn1 = loader.pop();
        process_weight(ffn1.weight, model.ffn_down_w[i], model.n_interm, n_embd);
        fill_span(model.ffn_down_b[i], ffn1.bias);

        // 8. LayerNorm2 -> ln2
        auto ln2 = loader.pop();
        fill_span(model.ln2_w[i], ln2.weight);
        fill_span(model.ln2_b[i], ln2.bias);
    }
    
    // 9. Classifier Head
    auto classifier = loader.pop();
    
    // Transpose [num_labels, embd] -> [embd, num_labels]
    auto process_weight_cls = [&](const std::vector<u64>& src, span<u64>& dest, int rows, int cols) {
         if (src.size() != rows * cols) {
             std::cerr << "[Error] process_weight_cls size mismatch! Expected " << rows * cols << ", got " << src.size() << std::endl;
             exit(1);
         }
         #pragma omp parallel for collapse(2)
         for (int r = 0; r < rows; r++) {
             for (int c = 0; c < cols; c++) {
                 dest[r * cols + c] = (u64)src[c * rows + r];
             }
         }
    };
    process_weight_cls(classifier.weight, model.classifier_w, model.n_embd, model.num_labels);
    fill_span(model.classifier_b, classifier.bias);
}


void test_acc_dealer(int pid, BertModel &model)
{
    init::from_id(pid);
    share_model_weights(model);

    span<u64> x(128 * model.n_embd);
    span<u32> mask(128);  // Attention mask

    input::call(x, CLIENT);
    input::call(mask, CLIENT);

    auto y = inference(x, model, mask);

    output::call(y);
    finalize::call();
}

void test_acc(int pid, BertModel &model, SharkLoader &input_loader, const int batch_count, std::string shark_dataset)
{
    // Sync number of batches
    if (pid == CLIENT) {
        int batch_count_check = input_loader.buffer.size();
        always_assert(batch_count == batch_count_check);
    }

    std::cout << "[INFO] Total batches to process: " << batch_count << std::endl;

    int idx = 0;
    // Iterate inputs
    for (int i = 0; i < batch_count; ++i)
    {
        init::from_id(pid);
        // Share weights
        SharkLoader weight_loader;
        if (pid == SERVER) {
            std::cout << "[INFO] (SERVER) Loading model weights..." << std::endl;
            weight_loader = SharkLoader(shark_dataset, "Bert_base", false);

            std::cout << "[INFO] Loading weights into model struct..." << std::endl;
            load_model_weights(model, weight_loader);
        }
        share_model_weights(model);
        std::cout << "[INFO] Model weights shared." << std::endl;

        span<u64> x(128 * model.n_embd);
        span<u32> mask(128);  // Attention mask (optional)
        size_t mask_size = 128;

        if (party == CLIENT) {
            auto input_pair = input_loader.pop();

            // Input is vector<u64> (reconstructed). Convert to span<T>
            if (input_pair.data.size() != 128 * model.n_embd) {
                 std::cerr << "[Error] Input size mismatch: " << input_pair.data.size() << std::endl;
                 exit(1);
            }
            fill_span(x, input_pair.data);

            // Load attention mask if available
            if (input_pair.has_mask) {
                mask_size = input_pair.attention_mask.size();
                mask = span<u32>(mask_size);
                fill_span(mask, input_pair.attention_mask);
                std::cout << "[DEBUG] Sample " << i << " has attention mask, size=" << mask_size << std::endl;
            } else
            {
                for (size_t i = 0; i < mask.size(); i++)
                {
                    mask[i] = 0;
                }
                std::cout << "[DEBUG] Sample " << i << " do not have attention mask, padding zeros" << std::endl;
            }

            // Then share input x
            input::call(x, CLIENT);
            input::call(mask, CLIENT);
        } else {
            input::call(x, CLIENT);
            mask = span<u32>(128);  // Default mask size matches n_token
            input::call(mask, CLIENT);
        }

        utils::start_timer("bert_idx_" + std::to_string(idx));
        auto y = inference(x, model, mask);
        utils::stop_timer("bert_idx_" + std::to_string(idx));

        output::call(y);
        shark::protocols::finalize::call();

        if (party == SERVER) { // Party 0 writes output
            std::ofstream out("bert_output_p0.txt", std::ios::app);
            for(size_t k=0; k<y.size(); ++k) {
                // Convert fixed point to float
                double val = fix2double(y[k]);
                out << val << " ";
            }
            out << std::endl;
            out.close();
        }

        std::cout << "Result " << idx << " computed." << std::endl;

        idx++;
    }
}


int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    std::cout << "[INFO] Starting program..." << std::endl;
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party_id> [dataset]" << std::endl;
        return 1;
    }
    
    int pid = std::atoi(argv[1]);
    std::string shark_dataset = "RTE";
    if (argc > 2) shark_dataset = argv[2];
    
    // Optional Model Config
    int n_layers = 12;
    int n_heads = 12;
    int n_embd = 768;
    int n_interm = 3072;
    int max_samples = 1;
    
    // Parse extra args if present (simple positional: layers heads embd interm)
    if (argc > 3) n_layers = std::atoi(argv[3]);
    if (argc > 4) n_heads = std::atoi(argv[4]);
    if (argc > 5) n_embd = std::atoi(argv[5]);
    if (argc > 6) n_interm = std::atoi(argv[6]);
    int num_labels = 2;
    if (argc > 7) num_labels = std::atoi(argv[7]);
    if (argc > 8) max_samples = std::atoi(argv[8]);

    std::cout << "[INFO] Network initialized. Dataset: " << shark_dataset << std::endl;
    std::cout << "[INFO] Config: L=" << n_layers << " H=" << n_heads << " D=" << n_embd << " I=" << n_interm << " Labels=" << num_labels << std::endl;

    if (pid == SERVER) {
        std::remove("bert_output_p0.txt");
    }

    std::cout << "[INFO] Allocating model..." << std::endl;
    BertModel model(n_layers, n_heads, n_embd, n_interm, num_labels, true);

    if (pid != DEALER)
    {
        std::cout << "[INFO] Loading shark data..." << std::endl;

        SharkLoader i_loader;
        if (pid == CLIENT) {
            std::cout << "[INFO] (CLIENT) Loading inputs..." << std::endl;
            i_loader = SharkLoader(shark_dataset, "Bert_base", true);
        }
        
        std::cout << "[INFO] Starting test_acc..." << std::endl;
        test_acc(pid, model, i_loader, max_samples, shark_dataset);
    }
    else
    {
        test_acc_dealer(pid, model);
    }

    return 0;
}
