#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/globals.hpp>


using namespace shark;
using namespace shark::protocols;

u32 double2fix(double x) {
    return static_cast<u32>(static_cast<std::make_signed_t<u32>>(x * (1 << FLOAT_PRECISION_32)));  // Note: must truncate the decimal part using static_cast<int_t>
}

double fix2double(u32 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u32>>(x)) / (1 << FLOAT_PRECISION_32);
}

double fix2double(u64 x) {
    return static_cast<double>(static_cast<std::make_signed_t<u64>>(x)) / (1 << FLOAT_PRECISION_64);
}

void test_main()
{
    // u64 n1 = 128;
    u64 n1 = 1;
    // u64 n2 = 128;
    u64 n2 = 10;
    u64 n = n1 * n2;
    shark::span<u32> X(n);

    if (party == SERVER) {
        for (int i = 0; i < n; ++i)
        {
            X[i] = double2fix(double(-1) - 4.0*double(i)/10.0);
            // X[i] = double2fix(i);
        }
    }

    input::call(X, 0);
    shark::utils::start_timer("softmax");
    auto Y = softmax::call(n1, n2, X);
    shark::utils::stop_timer("softmax");
    output::call(Y);
    finalize::call();
    utils::print_all_timers();
    if (party == SERVER)
    {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            std::cout << fix2double(Y[i]) << " \n";
            sum += fix2double(Y[i]);
        }
        std::cout << "SUM: " << sum << " \n";
    }
}


int main(int argc, char **argv)
{
    init::from_args(argc, argv);

    test_main();
}
