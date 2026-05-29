#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include <cryptoTools/Common/Defines.h>

#include <shark/types/u64.hpp>
#include <shark/types/u32.hpp>
#include <shark/types/u16.hpp>
#include <shark/types/u8.hpp>

namespace shark {
    typedef osuCrypto::block block;
    typedef __uint128_t u128;

    inline u64 getLow(const u128 &x)
    {
        return (u64)x;
    }

    inline u32 getLow32(const u128 &x)
    {
        return (u32)x;
    }

    inline u16 getLow16(const u128 &x)
    {
        return (u16)x;
    }

    inline void setLow(u128 &x, u64 low)
    {
        // ((u64*)&x)[0] = low;
        x = (x & (u128(-1) << 64)) | u128(low);
    }

    inline void setLow32(u128 &x, u32 low)
    {
        // ((u32*)&x)[0] = low;
        x = (x & (u128(-1) << 32)) | u128(low);
    }

    inline void setLow16(u128 &x, u16 low)
    {
        // ((u16*)&x)[0] = low;
        x = (x & (u128(-1) << 16)) | u128(low);
    }

    inline u64 getHigh(const u128 &x)
    {
        // return ((u64*)&x)[1];
        return u64(x >> 64);
    }

    inline u32 getHigh32(const u128 &x)
    {
        // return ((u32*)&x)[1];
        return u32(x >> 32);
    }

    inline u16 getHigh16(const u128 &x)
    {
        // return ((u16*)&x)[1];
        return u16(x >> 16);
    }

    inline void setHigh(u128 &x, u64 high)
    {
        // ((u64*)&x)[1] = high;
        x = (x & u128((u128(1) << 64) - 1)) | u128(high) << 64;
    }

    inline void setHigh32(u128 &x, u64 high)
    {
        // ((u32*)&x)[1] = high;
        x = (x & u128((u128(1) << 32) - 1)) | u128(high) << 32;
    }

    inline void setHigh16(u128 &x, u64 high)
    {
        // ((u16*)&x)[1] = high;
        x = (x & u128((u128(1) << 16) - 1)) | u128(high) << 16;
    }
}
