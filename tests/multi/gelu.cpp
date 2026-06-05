#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/gelu.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/globals.hpp>


using namespace shark;
using namespace shark::protocols;


u16 double2fix(double x) {
    return static_cast<u16>(static_cast<std::make_signed_t<u16>>(x * (1 << FLOAT_PRECISION_16)));  // Note: must truncate the decimal part using static_cast<int_t>
}

double fix2double(u16 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u16>>(x)) / (1 << FLOAT_PRECISION_16);
}

void test_main()
{
    u64 n = 25;
    shark::span<u16> X(n);

    if (party == SERVER) {
        for (int i = 0; i < n; ++i)
        {
            X[i] = double2fix(double(-1) - 4.0*double(i)/10.0);
            // X[i] = double2fix(i);
        }
    }

    input::call(X, 0);
    shark::utils::start_timer("gelu");
    auto Y = gelu::call(X);
    shark::utils::stop_timer("gelu");
    output::call(Y);
    finalize::call();
    utils::print_all_timers();
    if (party == SERVER)
    {
        for (int i = 0; i < n; ++i)
        {
            std::cout << fix2double(Y[i]) << " " << GeLU(double(-1) - 4.0*double(i)/10.0) << " \n";
            // std::cout << fix2double(Y[i]) << " " << GeLU(double(i)) << " \n";
        }
    }
}


int main(int argc, char **argv)
{
    init::from_args(argc, argv);

    test_main();
}
