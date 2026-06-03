#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/down.hpp>
#include <shark/protocols/output.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/globals.hpp>

using namespace shark;
using namespace shark::protocols;

u64 double2fix_64(double x) {
    return static_cast<u64>(static_cast<std::make_signed_t<u64>>(x * (1 << FLOAT_PRECISION_64)));  // Note: must truncate the decimal part using static_cast<int_t>
}

u32 double2fix_32(double x) {
    return static_cast<u32>(static_cast<std::make_signed_t<u32>>(x * (1 << FLOAT_PRECISION_32)));  // Note: must truncate the decimal part using static_cast<int_t>
}

u16 double2fix_16(double x) {
    return static_cast<u16>(static_cast<std::make_signed_t<u16>>(x * (1 << FLOAT_PRECISION_16)));  // Note: must truncate the decimal part using static_cast<int_t>
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
    shark::span<u64> X(n);
    shark::span<u64> expected(n);
    // shark::span<u32> X(n);
    // shark::span<u32> expected(n);

    if (party == SERVER) {
        for (int i = 0; i < n; ++i)
        {
            X[i] = double2fix_64(double(-1) + 4.0*double(i)/10.0);
            // X[i] = double2fix_32(double(-1) + 4.0*double(i)/10.0);
            expected[i] = X[i];
        }
    }

    input::call(X, 0);
    // auto Y = down::call_64_32(X, FLOAT_PRECISION, FLOAT_PRECISION_32);
    // auto Y = down::call_64_16(X, FLOAT_PRECISION, FLOAT_PRECISION_16);
    // auto Y = down::call_32_16(X, FLOAT_PRECISION_32, FLOAT_PRECISION_16);
    auto Y = truncate::call_64_32(X, 0);
    output::call(Y);
    finalize::call();
    if (party == SERVER)
    {
        for (u64 i = 0; i < n; ++i)
        {
            std::cout << i << " " << std::endl;
            std::cout << fix2double(Y[i]) << " " << fix2double(expected[i]) << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    init::from_args(argc, argv);
    
    test_main();
}
