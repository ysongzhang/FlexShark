#include <shark/utils/assert.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/protocols/common.hpp>
#include <shark/crypto/dpfring.hpp>
#include <cryptoTools/Crypto/AES.h>

namespace shark
{
    namespace crypto
    {
        using namespace osuCrypto;

        const block notOneBlock = toBlock(~0, ~1);
        const block notThreeBlock = toBlock(~0, ~3);
        const block ZeroBlock = toBlock(0, 0);
        const block OneBlock = toBlock(0, 1);
        const block TwoBlock = toBlock(0, 2);
        const block ThreeBlock = toBlock(0, 3);
        const block pt[4] = {ZeroBlock, OneBlock, TwoBlock, ThreeBlock};

        const AES ak0(toBlock(11898517819052905481ull, 11033408630406285283ull));
        const AES ak1(toBlock(9326388341714895150ull, 10128647480701028797ull));
        const AES ak2(toBlock(9873267564496987689ull, 9447730648146089691ull));
        const AES ak3(toBlock(10184333836464677069ull, 9405075354919185247ull));
        const AES ak4(toBlock(9778003634410817943ull, 12187153955522203843ull));
        const AES ak5(toBlock(13266479233780938046ull, 13416694524136657701ull));

        inline u8 lsb(const block &b)
        {
            return _mm_cvtsi128_si64x(b) & 1;
        }

        void convert_dpf(const block &in, u128 &out_ring, u128 &out_tag)
        {
            block ct[2];
            // AES ak(in);
            // ak.ecbEncTwoBlocks(pt, ct);
            ct[0] = ak4.ecbEncBlock(in) ^ in;
            ct[1] = ak5.ecbEncBlock(in) ^ in;
            out_ring = *(u128 *)ct;
            out_tag = *(u128 *)(ct + 1);
        }

        template <typename T>
        std::pair<DPFRingKey, DPFRingKey> dpfring_gen(int bin, const T alpha)
        {
            u128 payload_ring = 1;
            u128 payload_tag = shark::protocols::ring_key;

            auto s = protocols::rand<std::array<block, 2>>();
            block si[2][2];

            shark::span<block> k0(bin + 1);
            shark::span<block> k1(bin + 1);

            s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);
            k0[0] = s[0];
            k1[0] = s[1];
            
            block ct[4];
            u64 alpha64 = static_cast<u64>(alpha);

            for (int i = 0; i < bin; ++i)
            {
                const u8 keep = static_cast<uint8_t>(alpha64 >> (bin - 1 - i)) & 1;
                auto a = toBlock(keep);

                auto ss0 = s[0] & notThreeBlock;
                auto ss1 = s[1] & notThreeBlock;

                // AES ak0(ss0);
                // AES ak1(ss1);
                si[0][0] = ak0.ecbEncBlock(ss0) ^ ss0;
                si[0][1] = ak1.ecbEncBlock(ss0) ^ ss0;

                si[1][0] = ak0.ecbEncBlock(ss1) ^ ss1;
                si[1][1] = ak1.ecbEncBlock(ss1) ^ ss1;

                auto ti0 = lsb(s[0]);
                auto ti1 = lsb(s[1]);
                u128 sign = 1;
                if (ti1 == 1) sign -= 2;

                std::array<block, 2> siXOR{si[0][0] ^ si[1][0], si[0][1] ^ si[1][1]};

                // get the left and right t_CW bits
                std::array<block, 2> t{
                    (OneBlock & siXOR[0]) ^ a ^ OneBlock,
                    (OneBlock & siXOR[1]) ^ a};

                // take scw to be the bits [127, 2] as scw = s0_loss ^ s1_loss
                auto scw = siXOR[keep ^ 1] & notThreeBlock;

                k0[i + 1] = k1[i + 1] = scw           // set bits [127, 2] as scw = s0_loss ^ s1_loss
                                        ^ (t[0] << 1) // set bit 1 as tL
                                        ^ t[1];       // set bit 0 as tR

                auto si0Keep = si[0][keep];
                auto si1Keep = si[1][keep];

                // extract the t^Keep_CW bit
                auto TKeep = t[keep];

                // set the next level of s,t
                s[0] = si0Keep ^ (zeroAndAllOne[ti0] & (scw ^ TKeep));
                s[1] = si1Keep ^ (zeroAndAllOne[ti1] & (scw ^ TKeep));
            }

            u128 s0_converted_ring;
            u128 s0_converted_tag;
            u128 s1_converted_ring;
            u128 s1_converted_tag;
            // convert(Bout, groupSize, s[0] & notThreeBlock, s0_converted);
            convert_dpf(s[0] & notThreeBlock, s0_converted_ring, s0_converted_tag);
            // convert(Bout, groupSize, s[1] & notThreeBlock, s1_converted);
            convert_dpf(s[1] & notThreeBlock, s1_converted_ring, s1_converted_tag);

            // this block needs to be edited
            u128 g0_ring = s1_converted_ring - s0_converted_ring + payload_ring;
            u128 g0_tag = s1_converted_tag - s0_converted_tag + payload_tag;
            if (lsb(s[1]) == 1)
            {
                g0_ring = -g0_ring;
                g0_tag = -g0_tag;
            }

            return std::make_pair(
                DPFRingKey(k0, g0_ring, g0_tag),
                DPFRingKey(k1, g0_ring, g0_tag)
            );
        }

        template <typename T>
        std::tuple<u128, u128> do_leaf(int party, const DPFRingKey &key, block s, const std::vector<T> &lut, u64 x, T lut_offset, int bin)
        {
            u8 t = lsb(s);
            u128 s_converted_ring;
            u128 s_converted_tag;

            convert_dpf(s & notThreeBlock, s_converted_ring, s_converted_tag);

            if (t)
            {
                s_converted_ring += key.g_ring;
                s_converted_tag += key.g_tag;
            }

            if (party == 1)
            {
                s_converted_ring = -s_converted_ring;
                s_converted_tag = -s_converted_tag;
            }

            u64 lut_offset_64 = static_cast<u64>(lut_offset);
            u64 idx = (x + lut_offset_64) % (1ull << bin);
            return std::make_tuple(s_converted_ring * lut[idx], s_converted_tag * lut[idx]);
        }

        template <typename T>
        std::tuple<u128, u128> do_subtree(int party, const DPFRingKey &key, const std::vector<T> &lut, int i, int bin, u64 curr_x, block s, T lut_offset)
        {
            if (i == bin)
            {
                return do_leaf(party, key, s, lut, curr_x, lut_offset, bin);
            }

            u8 t_previous = lsb(s);
            block cw = _mm_loadu_si128(key.k.data() + i + 1);

            const auto scw = (cw & notThreeBlock);
            block ds[] = { ((cw >> 1) & OneBlock), (cw & OneBlock) };
            const auto mask = zeroAndAllOne[t_previous];
            auto ss = s & notThreeBlock;

            u128 out_ring = 0;
            u128 out_tag = 0;

            for (int keep = 0; keep < 2; ++keep)
            {
                block ct;
                if (keep == 0)
                {
                    ct = ak0.ecbEncBlock(ss) ^ ss;
                }
                else
                {
                    ct = ak1.ecbEncBlock(ss) ^ ss;
                }

                block stcw = ((scw ^ ds[keep]) & mask) ^ ct;
                auto tup = do_subtree(party, key, lut, i + 1, bin, (curr_x << 1) + keep, stcw, lut_offset);
                out_ring += std::get<0>(tup);
                out_tag += std::get<1>(tup);
            }

            return std::make_tuple(out_ring, out_tag);
        }

        template <typename T>
        std::tuple<u128, u128> dpfring_evalall_reduce(int party, const DPFRingKey &key, const std::vector<T> &lut, T lut_offset)
        {
            int bin = key.k.size() - 1;
            always_assert(lut.size() == (1ull<<bin));
        
            return do_subtree(party, key, lut, 0, bin, 0, _mm_loadu_si128(key.k.data()), lut_offset);
        }

        template std::pair<DPFRingKey, DPFRingKey> dpfring_gen<u64>(int, const u64);
        template std::pair<DPFRingKey, DPFRingKey> dpfring_gen<u32>(int, const u32);
        template std::pair<DPFRingKey, DPFRingKey> dpfring_gen<u16>(int, const u16);

        template std::tuple<u128, u128> dpfring_evalall_reduce<u64>(int, const DPFRingKey &, const std::vector<u64> &, u64);
        template std::tuple<u128, u128> dpfring_evalall_reduce<u32>(int, const DPFRingKey &, const std::vector<u32> &, u32);
        template std::tuple<u128, u128> dpfring_evalall_reduce<u16>(int, const DPFRingKey &, const std::vector<u16> &, u16);


        std::tuple<std::tuple<u128, u128>, std::tuple<u128, u128>> do_leaf(int party, const DPFRingKey &key, block s, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, u64 x, u32 lut_offset, int bin)
        {
            u8 t = lsb(s);
            u128 s_converted_ring;
            u128 s_converted_tag;

            convert_dpf(s & notThreeBlock, s_converted_ring, s_converted_tag);

            if (t)
            {
                s_converted_ring += key.g_ring;
                s_converted_tag += key.g_tag;
            }

            if (party == 1)
            {
                s_converted_ring = -s_converted_ring;
                s_converted_tag = -s_converted_tag;
            }

            u64 lut_offset_64 = static_cast<u64>(lut_offset);
            u64 idx = (x + lut_offset_64) % (1ull << bin);
            auto res1 = std::make_tuple(s_converted_ring * lut_1[idx], s_converted_tag * lut_1[idx]);
            auto res2 = std::make_tuple(s_converted_ring * lut_2[idx], s_converted_tag * lut_2[idx]);
            return std::make_tuple(res1, res2);
        }

        std::tuple<std::tuple<u128, u128>, std::tuple<u128, u128>> do_subtree(int party, const DPFRingKey &key, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, int i, int bin, u64 curr_x, block s, u32 lut_offset)
        {
            if (i == bin)
            {
                return do_leaf(party, key, s, lut_1, lut_2, curr_x, lut_offset, bin);
            }

            u8 t_previous = lsb(s);
            block cw = _mm_loadu_si128(key.k.data() + i + 1);

            const auto scw = (cw & notThreeBlock);
            block ds[] = { ((cw >> 1) & OneBlock), (cw & OneBlock) };
            const auto mask = zeroAndAllOne[t_previous];
            auto ss = s & notThreeBlock;

            u128 out_ring_1 = 0;
            u128 out_tag_1 = 0;
            u128 out_ring_2 = 0;
            u128 out_tag_2 = 0;

            for (int keep = 0; keep < 2; ++keep)
            {
                block ct;
                if (keep == 0)
                {
                    ct = ak0.ecbEncBlock(ss) ^ ss;
                }
                else
                {
                    ct = ak1.ecbEncBlock(ss) ^ ss;
                }

                block stcw = ((scw ^ ds[keep]) & mask) ^ ct;
                auto tup = do_subtree(party, key, lut_1, lut_2, i + 1, bin, (curr_x << 1) + keep, stcw, lut_offset);

                auto &t1 = std::get<0>(tup);
                auto &t2 = std::get<1>(tup);

                out_ring_1 += std::get<0>(t1);
                out_tag_1  += std::get<1>(t1);

                out_ring_2 += std::get<0>(t2);
                out_tag_2  += std::get<1>(t2);
            }

            return std::make_tuple(std::make_tuple(out_ring_1, out_tag_1), std::make_tuple(out_ring_2, out_tag_2));
        }

        std::tuple<std::tuple<u128, u128>, std::tuple<u128, u128>> dpfring_evalall_reduce(int party, const DPFRingKey &key, const std::vector<u32> &lut_1, const std::vector<u32> &lut_2, u32 lut_offset)
        {
            int bin = key.k.size() - 1;
            always_assert(lut_1.size() == (1ull<<bin));
            always_assert(lut_2.size() == (1ull<<bin));
        
            return do_subtree(party, key, lut_1, lut_2, 0, bin, 0, _mm_loadu_si128(key.k.data()), lut_offset);
        }
    }
}
