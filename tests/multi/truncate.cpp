#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/output.hpp>
#include <shark/utils/timer.hpp>

using namespace shark;
using namespace shark::protocols;

// using base_t = u64;
// constexpr int BITLEN = 64;
// constexpr int FLOAT_PRECISION = 16;
using base_t = u32;
constexpr int BITLEN = 32;
constexpr int FLOAT_PRECISION = 6;
constexpr int truncateValue = 1 << FLOAT_PRECISION;

base_t double2fix(double x) {

    return static_cast<base_t>(static_cast<std::make_signed_t<base_t>>(x * truncateValue));  // Note: must truncate the decimal part using static_cast<int_t>
}

void print_bits(base_t x)
{
    for (int i = (BITLEN - 1); i >= 0; --i)
    {
        std::cout << ((x >> i) & base_t(1));
    }
    std::cout << '\n';
}

void test_main()
{
    u64 n = 10;
    shark::span<base_t> X(n);
    shark::span<base_t> expected(n);

    if (party == SERVER) {
        for (int i = 0; i < n; ++i)
        {
            X[i] = double2fix(double(-1) + 4.0*double(i)/10.0);
            // X[i] = 4096 * 4096 + i + (base_t(1) << (BITLEN - 1)) + (base_t(1) << (BITLEN - 2)); // test negative numbers as well 
            expected[i] = X[i];
            // X[i] = double2fix(i);
        }
    }

    input::call(X, 0);
    utils::start_timer("trunc");
    auto Y = truncate::call(X, FLOAT_PRECISION);
    utils::stop_timer("trunc");
    output::call(Y);
    finalize::call();
    if (party == SERVER)
    {
        for (u64 i = 0; i < n; ++i)
        {
            // std::cout << Y[i] << " " << expected[i] << std::endl;
            std::cout << i << " " << std::endl;
            print_bits(expected[i]);
            print_bits(Y[i]);
        }
    }
    // utils::print_all_timers();
}

int main(int argc, char **argv)
{
    init::from_args(argc, argv);
    
    test_main();
}
