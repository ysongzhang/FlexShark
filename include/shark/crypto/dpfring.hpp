#pragma once

#include <tuple>
#include <vector>
#include <shark/types/span.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/u32.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace crypto
    {
        struct DPFRingKey
        {
            shark::span<block> k;
            u128 g_ring;
            u128 g_tag;

            DPFRingKey(const shark::span<block> &k, u128 g_ring, u128 g_tag)
                : k(k), g_ring(g_ring), g_tag(g_tag)
            {
            }

            DPFRingKey(shark::span<block> &&k, u128 g_ring, u128 g_tag)
                : k(std::move(k)), g_ring(g_ring), g_tag(g_tag)
            {
            }

            // move constructor
            DPFRingKey(DPFRingKey &&other) : k(std::move(other.k)), g_ring(other.g_ring), g_tag(other.g_tag)
            {
            }

            // move assignment
            DPFRingKey &operator=(DPFRingKey &&other)
            {
                k = std::move(other.k);
                g_ring = other.g_ring;
                g_tag = other.g_tag;
                return *this;
            }

            DPFRingKey() = default;
        };

        
        template <typename T>
        std::pair<DPFRingKey, DPFRingKey> dpfring_gen(int bin, const T alpha);

        template <typename T>
        std::tuple<u128, u128> dpfring_evalall_reduce(int party, const DPFRingKey &key, const std::vector<T> &lut, T lut_offset);

        std::tuple<std::tuple<u128, u128>, std::tuple<u128, u128>> dpfring_evalall_reduce(int party, const DPFRingKey &key, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, u32 lut_offset);
    }
}