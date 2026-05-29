#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/b2a.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

using u64 = shark::u64;
using u32 = shark::u32;
using u16 = shark::u16;
using u8 = shark::u8;
using namespace shark::protocols;

int main(int argc, char **argv)
{
    init::from_args(argc, argv);
    
    u64 n = 10000;
    // shark::span<u64> X(n);
    // shark::span<u32> X(n);
    shark::span<u16> X(n);
    shark::span<u8> d_expected(n);

    if (party == SERVER) {
        for (u64 i = 0; i < n; ++i)
        {
            // X[i] = rand();
            X[i] = rand() % 32768;

            d_expected[i] = rand() % 2;
            if (d_expected[i] == 0)
                X[i] = -X[i];
        }
    }

    input::call(X, 0);
    shark::utils::start_timer("drelu");
    auto Y = drelu::call(X);
    auto Z = b2a::call_16(Y);
    shark::utils::stop_timer("drelu");
    output::call(Z);
    finalize::call();
    if (party == SERVER)
    {
        for (u64 i = 0; i < n; ++i)
        {
            // std::cout << (int)Z[i] << " " << (int)d_expected[i] << std::endl;
            always_assert(Z[i] == (u16)d_expected[i]);
        }
    }
    shark::utils::print_timer("drelu");
}
