#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/up.hpp>
#include <shark/protocols/output.hpp>
#include <shark/utils/timer.hpp>

using namespace shark;
using namespace shark::protocols;

constexpr int FLOAT_PRECISION_64 = 16;
constexpr int FLOAT_PRECISION_32 = 12;
constexpr int FLOAT_PRECISION_16 = 6;

u64 double2fix_64(double x) {
    return static_cast<u64>(static_cast<std::make_signed_t<u64>>(x * (1 << FLOAT_PRECISION_64)));
}

u32 double2fix_32(double x) {
    return static_cast<u32>(static_cast<std::make_signed_t<u32>>(x * (1 << FLOAT_PRECISION_32)));
}

u16 double2fix_16(double x) {
    return static_cast<u16>(static_cast<std::make_signed_t<u16>>(x * (1 << FLOAT_PRECISION_16)));
}

double fix2double(u64 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u64>>(x)) / (1 << FLOAT_PRECISION_64);
}

double fix2double(u32 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u32>>(x)) / (1 << FLOAT_PRECISION_32);
}

double fix2double(u16 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u16>>(x)) / (1 << FLOAT_PRECISION_16);
}

void test_main()
{
    u64 n = 10;

    // ---- Test 1: u16 -> u32 ----
    {
        shark::span<u16> X(n);
        shark::span<u16> expected(n);

        if (party == SERVER) {
            for (int i = 0; i < n; ++i)
            {
                X[i] = double2fix_16(double(-1) + 4.0 * double(i) / 10.0);
                expected[i] = X[i];
            }
        }

        input::call(X, 0);
        auto Y = up::call_16_32(X, FLOAT_PRECISION_16, FLOAT_PRECISION_32);
        output::call(Y);

        if (party == SERVER)
        {
            std::cout << "=== up 16 -> 32 ===" << std::endl;
            for (u64 i = 0; i < n; ++i)
            {
                std::cout << i << ": " << fix2double(Y[i]) << " (expected " << fix2double(expected[i]) << ")" << std::endl;
            }
        }
    }

    // ---- Test 2: u32 -> u64 ----
    {
        shark::span<u32> X(n);
        shark::span<u32> expected(n);

        if (party == SERVER) {
            for (int i = 0; i < n; ++i)
            {
                X[i] = double2fix_32(double(-1) + 4.0 * double(i) / 10.0);
                expected[i] = X[i];
            }
        }

        input::call(X, 0);
        auto Y = up::call_32_64(X, FLOAT_PRECISION_32, FLOAT_PRECISION_64);
        output::call(Y);

        if (party == SERVER)
        {
            std::cout << "=== up 32 -> 64 ===" << std::endl;
            for (u64 i = 0; i < n; ++i)
            {
                std::cout << i << ": " << fix2double(Y[i]) << " (expected " << fix2double(expected[i]) << ")" << std::endl;
            }
        }
    }

    // ---- Test 3: u16 -> u64 ----
    {
        shark::span<u16> X(n);
        shark::span<u16> expected(n);

        if (party == SERVER) {
            for (int i = 0; i < n; ++i)
            {
                X[i] = double2fix_16(double(-1) + 4.0 * double(i) / 10.0);
                expected[i] = X[i];
            }
        }

        input::call(X, 0);
        auto Y = up::call_16_64(X, FLOAT_PRECISION_16, FLOAT_PRECISION_64);
        output::call(Y);

        if (party == SERVER)
        {
            std::cout << "=== up 16 -> 64 ===" << std::endl;
            for (u64 i = 0; i < n; ++i)
            {
                std::cout << i << ": " << fix2double(Y[i]) << " (expected " << fix2double(expected[i]) << ")" << std::endl;
            }
        }
    }

    finalize::call();
}

int main(int argc, char **argv)
{
    init::from_args(argc, argv);
    test_main();
}
