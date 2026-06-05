#include <shark/utils/lookuptable.hpp>

namespace shark
{
    std::vector<u32> lut_inv;
    int bin_inv;
    std::vector<u32> lut_exp_2;
    int bin_exp_2;
    std::vector<u32> lut_exp_b;
    int bin_exp_b;
    std::vector<u32> lut_rsqrt;
    int bin_rsqrt;
    std::vector<u32> lut_exsqrt;
    int bin_exsqrt;
    std::vector<u16> lut_gelu;
    int bin_gelu;

    void init_luts()
    {
        // Init inv
        u32 table_size_inv = (u32)ceil((1.0  - 1.0 / SCALE_BASE) * (1u << FLOAT_PRECISION_32));
        bin_inv = ceil_log2_u32(table_size_inv);
        u32 padding_size_inv = 1u << bin_inv;
        lut_inv.resize(padding_size_inv);
        for (u32 i = 0; i < table_size_inv; ++i)
        {
            float real_value = float(i) / (1u << FLOAT_PRECISION_32) + 1.0 / SCALE_BASE;
            lut_inv[i] = u32((1.0 / real_value) * (1u << FLOAT_PRECISION_32));
        }

        // Init exp_2
        u32 table_size_exp_2 = FLOAT_PRECISION_32 + 2;
        bin_exp_2 = ceil_log2_u32(table_size_exp_2);
        u32 padding_size_exp_2 = 1u << bin_exp_2;
        lut_exp_2.resize(padding_size_exp_2);
        for (u32 i = 0; i < table_size_exp_2; ++i)
        {
            if (i <= FLOAT_PRECISION_32)  // [0,f]
            {
                lut_exp_2[i] = u32(1u << (FLOAT_PRECISION_32 - i));
            }
            else  // f+1
            {
                lut_exp_2[i] = 0u;
            }
        }

        // Init exp_b
        u32 table_size_exp_b = u32((int)(log(pow(2, 2 * FLOAT_PRECISION_32)) / log(SCALE_BASE)) + 1);
        bin_exp_b = ceil_log2_u32(table_size_exp_b);
        u32 padding_size_exp_b = 1u << bin_exp_b;
        lut_exp_b.resize(padding_size_exp_b);
        for(int i = 0; i < table_size_exp_b; i++)
        {
            lut_exp_b[i] = u32((pow(2, FLOAT_PRECISION_32) / pow(SCALE_BASE, i)) * (1 << FLOAT_PRECISION_32));
        }

        // Init rsqrt
        u32 table_size_rsqrt = (u32)ceil((1.0  - 1.0 / SCALE_BASE) * (1 << FLOAT_PRECISION_32));
        bin_rsqrt = ceil_log2_u32(table_size_rsqrt);
        u32 padding_size_rsqrt = 1u << bin_rsqrt;
        lut_rsqrt.resize(padding_size_rsqrt);
        for(int i = 0; i < table_size_rsqrt; i++)
        {
            float real_value = float(i) / (1 << FLOAT_PRECISION_32) + 1.0 / SCALE_BASE;
            lut_rsqrt[i] = u32(sqrt(1 / real_value) * (1 << FLOAT_PRECISION_32));
        }

        // Init exsqrt
        u32 table_size_exsqrt = (u32)((int)(log(pow(2, 2 * FLOAT_PRECISION_32)) / log(SCALE_BASE)) + 1);
        bin_exsqrt = ceil_log2_u32(table_size_exsqrt);
        u32 padding_size_exsqrt = 1u << bin_exsqrt;
        lut_exsqrt.resize(padding_size_exsqrt);
        for(int i = 0; i < table_size_exsqrt; i++)
        {
            lut_exsqrt[i] = u32(sqrt(pow(2, FLOAT_PRECISION_32) / pow(SCALE_BASE, i)) * (1 << FLOAT_PRECISION_32));
        }
        
        // Init gelu
        u32 table_size_gelu = 4 * (1 << FLOAT_PRECISION_16);
        bin_gelu = ceil_log2_u32(table_size_gelu);
        u32 padding_size_gelu = 1u << bin_gelu;
        lut_gelu.resize(padding_size_gelu);
        for (int i = 0; i < table_size_gelu; i++)
        {
            float real_value = float(i) / (1 << FLOAT_PRECISION_16);
            lut_gelu[i] = u16(int16_t((ReLU(real_value) - GeLU(real_value)) * (1 << FLOAT_PRECISION_16)));
        }
    }
}