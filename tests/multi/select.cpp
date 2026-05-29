#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/select.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

using namespace shark;
using namespace shark::protocols;

int main(int argc, char **argv)
{
    init::from_args(argc, argv);
    
    u64 n = 10000;
    // shark::span<u64> X(n);
    // shark::span<u32> X(n);
    shark::span<u16> X(n);
    shark::span<u8> d(n);
    // shark::span<u64> Z(n);
    // shark::span<u32> Z(n);
    shark::span<u16> Z(n);

    if (party == SERVER) {
        for (u64 i = 0; i < n; ++i)
        {
            X[i] = rand();
            d[i] = rand() % 2;
            Z[i] = d[i] * X[i];
        }
    }

    input::call(X, 0);
    input::call(d, 0);
    shark::utils::start_timer("select");
    auto Y = select::call(d, X);
    shark::utils::stop_timer("select");
    output::call(Y);
    finalize::call();
    if (party == SERVER)
    {
        for (u64 i = 0; i < n; ++i)
        {
            always_assert(Y[i] == Z[i]);
        }
    }
    shark::utils::print_timer("select");
}
