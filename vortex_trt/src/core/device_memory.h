/*
Wrapper of GPU memory
*/
#pragma once

#include "cuda_utils.h"

namespace vortex
{
    class DeviceMemory
    {
    private:
        float* m_Data;
        
    public:
        DeviceMemory(uint32_t size)
        {

        }

        ~DeviceMemory()
        {

        }

        float* Ptr() const { return m_Data; }

        // void CopyFromHost();
        // void CopyFromDevice();
        // void CopyToHost();
        // void CopyToDevice();
    };
}
