#include <shark/protocols/finalize.hpp>

namespace shark
{
    namespace protocols
    {
        namespace finalize
        {
            void gen()
            {
                server->close();
                client->close();
            }

            void eval()
            {
                dealer->close();
                peer->close();
            }

            void call()
            {
                if (party == DEALER)
                {
                    gen();
                }
                else
                {
                    eval();
                }
            }

            void refresh_preprocessing(bool oneShot)
            {
                if (party == DEALER)
                {
                }
                else
                {
                    dealer->close();
                    dealer = new Dealer(filename[party], oneShot);
                    ring_key = dealer->recv<u64>();
                    bit_key = dealer->recv<u64>();
                }
            }
        }
        
    }
    
}
