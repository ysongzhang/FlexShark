#include "bert_model.hpp"

using namespace shark;
using namespace shark::protocols;

void test_base(shark::span<u64> &in, BertModel &model)
{
    utils::start_timer("input");
    input::call(in, CLIENT);
    share_model_weights(model);
    utils::stop_timer("input");

    utils::start_timer("bert");
    auto y = inference(in, model);
    utils::stop_timer("bert");

    output::call(y);

    finalize::call();
    utils::print_all_timers();
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    init::from_args(argc, argv);

    u64 n_token = 1;
    
    BertModel model;
    span<u64> x(n_token * model.n_embd);
    test_base(x, model);
}
