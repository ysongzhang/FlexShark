#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/nexp.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>


using namespace shark;
using namespace shark::protocols;

u32 double2fix(double x) {
    return static_cast<u32>(static_cast<std::make_signed_t<u32>>(x * (1 << FLOAT_PRECISION_32)));  // Note: must truncate the decimal part using static_cast<int_t>
}

double fix2double(u32 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u32>>(x)) / (1 << FLOAT_PRECISION_32);
}

void test_main()
{
    u64 n = 10;
    shark::span<u32> X(n);

    if (party == SERVER) {
        for (int i = 0; i < n; ++i)
        {
            X[i] = double2fix(-1.0 * i);
            // std::cout << X[i] << "\n";
        }
    }

    input::call(X, 0);
    shark::utils::start_timer("nexp");
    auto Y = nexp::call(X);
    shark::utils::stop_timer("nexp");
    output::call(Y);
    finalize::call();
    utils::print_all_timers();
    if (party == SERVER)
    {
        for (int i = 0; i < n; ++i)
        {
            std::cout << fix2double(Y[i]) << " " << exp(-1.0 * i) << "\n";
        }
    }
}

int main(int argc, char **argv)
{
    init::from_args(argc, argv);

    test_main();
}
