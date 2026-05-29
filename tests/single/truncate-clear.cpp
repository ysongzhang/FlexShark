#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

using namespace shark;

// using base_t = u64;
// constexpr int BITLEN = 64;
// constexpr int FLOAT_PRECISION = 16;

using base_t = u32;
constexpr int BITLEN = 32;
constexpr int FLOAT_PRECISION = 12;

void print_bits(base_t x)
{
    for (int i = (BITLEN - 1); i >= 0; --i)
    {
        std::cout << ((x >> i) & 1ULL);
    }
    std::cout << '\n';
}

int main(int argc, char **argv)
{
    int f = FLOAT_PRECISION;
    int bw = BITLEN;

    // for (int i = 0; i < 10; ++i)
    // {
    //     // y = x + 2^{k-2}
    //     // LRS(y) = LRS(m) + LRS(r) - 2^{k-f} * (MSB(m) + MSB(r) - MSB(m) * MSB(r))
    //     // ARS(x) = LRS(y) - 2^{k-f-2}
    //     u64 x = double2fix(double(-1) - 4.0*double(i)/10.0);
    //     // u64 x = double2fix(i);
    //     u64 y = x + u64(1ULL << (bw - 2));
    //     u64 r = double2fix(2.0);
    //     u64 m = y - r;
    //     u64 m_msb = (m >> (bw - 1)) & 1ull;
    //     u64 r_msb = (r >> (bw - 1)) & 1ull;
    //     u64 y_trunc = (m >> f) + (r >> f) - u64(1ULL << (bw - f)) * (m_msb + r_msb - m_msb * r_msb);
    //     u64 x_trunc = y_trunc - u64(1ULL << (bw - f - 2));
    //     std::cout << "=================\n";
    //     print_bits(x);
    //     print_bits(x_trunc);
    // }

    // base_t x = 3825205248 + (base_t(1) << (bw - 1)) + (base_t(1) << (bw - 2));  // 228 * 4096 * 4096 which would be overflowed.
    base_t x = 933888 + (base_t(1) << (bw - 1)) + (base_t(1) << (bw - 2));
    // u64 x = double2fix(i);
    base_t y = x + (base_t(1) << (bw - 2));
    base_t r = 256;
    base_t m = y + r;
    base_t m_msb = (m >> (bw - 1)) & base_t(1);
    base_t r_msb = (r >> (bw - 1)) & base_t(1);
    base_t y_trunc = (m >> f) - (r >> f) + base_t(base_t(1) << (bw - f)) * (1 - m_msb) * r_msb;
    base_t x_trunc = y_trunc - base_t(base_t(1) << (bw - f - 2));
    std::cout << "=================\n";
    std::cout << x << " " << x_trunc << std::endl;
    print_bits(x);
    print_bits(x_trunc);
}