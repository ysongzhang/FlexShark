#pragma once

#include <shark/protocols/common.hpp>

namespace shark
{
    namespace protocols
    {
        namespace finalize
        {
            void gen();
            void eval();
            void call();
            void refresh_preprocessing(bool oneShot = true);
        }
        
    }
    
}
