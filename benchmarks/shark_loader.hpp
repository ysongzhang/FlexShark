#pragma once

#include <cnpy.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <set>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "shark/utils/globals.hpp"
#include "shark/types/u64.hpp"

using namespace shark;
using namespace shark::protocols;

// Helper to check file existence
inline bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// Helper to copy vector<u64> to span<T>
template <typename T>
void fill_span(span<T> &dest, const std::vector<u64> &src) {
    if (dest.size() != src.size()) {
        std::cerr << "[Error] fill_span size mismatch! Expected " << dest.size() << ", got " << src.size() << std::endl;
        exit(1);
    }
    for(size_t i=0; i<src.size(); ++i) dest[i] = (T)src[i];
}

class SharkLoader
{
public:
    struct ParamPair {
        std::vector<shark::u64> weight; // Reconstructed plaintext weights in fixed-point form
        std::vector<shark::u64> bias;
        bool has_bias = false;
        
        // Data-mode input vector
        std::vector<shark::u64> data;
        bool is_data = false;

        // Padding mask stored separately from the input tensor
        std::vector<shark::u64> attention_mask;
        bool has_mask = false;
    };

    std::deque<ParamPair> buffer;
    std::string shark_dataset;
    std::string shark_model;
    int party_id; // Present for compatibility with older loader flows

    SharkLoader() {}

    SharkLoader(const std::string &shark_dataset_arg, const std::string &shark_model_arg, bool is_data_mode) 
        : shark_dataset(shark_dataset_arg), shark_model(shark_model_arg)
    {
        std::map<std::string, std::vector<u64>> loaded_shares;
        
        std::string path;
        if (is_data_mode) {
            path = "log/data_shares/" + shark_dataset + ".npz";
        } else {
            path = "log/model_shares/" + shark_model + "_" + shark_dataset + ".npz";
        }
        
        if (!file_exists(path)) {
            std::cerr << "[Loader] Error: File not found: " << path << std::endl;
            exit(1);
        }
        
        std::cout << "[Loader] Loading " << path << "..." << std::endl;
        cnpy::npz_t npz = cnpy::npz_load(path);
        
        std::vector<std::string> sorted_keys;
        std::vector<std::string> mask_keys;
        for (auto &kv : npz) {
            std::string key = kv.first;
            if (is_data_mode) {
                // Data entries end in '0'; masks use the matching numeric prefix plus 'm'
                if (key.length() > 0 && key.back() == '0') {
                    sorted_keys.push_back(key);
                } else if (key.length() > 0 && key.back() == 'm') {
                    mask_keys.push_back(key);
                }
            } else {
                // Model entries use 0t for weights and 0s for biases
                if (key.length() > 1 && key.substr(key.length()-2) == "0t") sorted_keys.push_back(key);
                if (key.length() > 1 && key.substr(key.length()-2) == "0s") sorted_keys.push_back(key);
            }
        }
        std::sort(sorted_keys.begin(), sorted_keys.end());
        std::sort(mask_keys.begin(), mask_keys.end());

        for (const auto& key : sorted_keys) {
            loaded_shares[key] = load_vector(npz[key]);
        }
        for (const auto& key : mask_keys) {
            loaded_shares[key] = load_vector(npz[key]);
        }
        
        if (is_data_mode) {
            // Inputs and masks are paired by their shared numeric prefix
            size_t num_inputs = sorted_keys.size();
            size_t num_masks = mask_keys.size();

            std::cout << "[Loader] Data mode: " << num_inputs << " inputs, "
                      << num_masks << " masks" << std::endl;

            // Print a small preview of the discovered key ordering
            for (size_t i = 0; i < std::min(size_t(5), num_inputs); ++i) {
                std::cout << "[Loader] Input key " << i << ": " << sorted_keys[i];
                if (i < num_masks) {
                    std::cout << " -> Mask key: " << mask_keys[i];
                }
                std::cout << std::endl;
            }

            for (size_t i = 0; i < num_inputs; ++i) {
                ParamPair p;
                p.is_data = true;
                p.data = loaded_shares[sorted_keys[i]];

                if (i < num_masks) {
                    p.attention_mask = loaded_shares[mask_keys[i]];
                    p.has_mask = true;

                    // Sanity check that the input and mask prefixes still line up
                    std::string input_prefix = sorted_keys[i].substr(0, sorted_keys[i].length() - 1);
                    std::string mask_prefix = mask_keys[i].substr(0, mask_keys[i].length() - 1);
                    if (input_prefix != mask_prefix) {
                        std::cerr << "[WARNING] Key mismatch! Input: " << sorted_keys[i]
                                  << " vs Mask: " << mask_keys[i] << std::endl;
                    }
                }

                buffer.push_back(p);
            }
        } else {
            // Model mode expects the flat sequence W, B, W, B in order
            std::vector<std::string> types;
            std::vector<std::string> ordered_keys;
            for (const auto& key : sorted_keys) {
                std::string type = key.substr(key.length()-2);
                types.push_back(type);
                ordered_keys.push_back(key);
            }

            size_t num_keys = ordered_keys.size();
            for (size_t i = 0; i < num_keys; ++i) {
                const std::string& key = ordered_keys[i];
                const std::string& type = types[i];
                auto& arr = npz[key];
                std::vector<u64> data = load_vector(arr);

                if (type == "0t") {
                    ParamPair p;
                    p.weight = data;
                    buffer.push_back(p);
                } else if (type == "0s") {
                    if (buffer.empty()) {
                        std::cerr << "[Loader] Error: Bias without Weight at index " << i << std::endl;
                        exit(1);
                    }
                    buffer.back().bias = data;
                    buffer.back().has_bias = true;
                }
            }
        }
        
        std::cout << "[Loader] Loaded " << buffer.size() << " items." << std::endl;
    }

    std::vector<u64> load_vector(cnpy::NpyArray &arr)
    {
        std::vector<u64> raw_data;
        size_t num_elements = arr.num_bytes() / arr.word_size;
        raw_data.reserve(num_elements);

        if (arr.word_size == 8)
        {
            // Most exported arrays are already stored as 64-bit values
            u64* loaded = arr.data<u64>();
            raw_data.assign(loaded, loaded + num_elements);
        }
        else if (arr.word_size == 4)
        {
            // Promote 32-bit arrays to the loader's uniform u64 representation
            uint32_t* loaded = arr.data<uint32_t>();
            for(size_t k=0; k<num_elements; ++k) raw_data.push_back((u64)loaded[k]);
        }
        else
        {
            std::cerr << "[Loader] Error: Unsupported word size " << arr.word_size << std::endl;
            exit(1);
        }
        return raw_data;
    }

    ParamPair pop()
    {
        if (buffer.empty()) {
            std::cerr << "[Loader] Error: Buffer empty when requesting parameters!" << std::endl;
            exit(1);
        }
        ParamPair p = buffer.front();
        buffer.pop_front();
        return p;
    }
    
    void skip(int n) {
        for(int i=0; i<n; ++i) {
            if(!buffer.empty()) buffer.pop_front();
        }
        std::cout << "[Loader] Skipped " << n << " items." << std::endl;
    }
};
