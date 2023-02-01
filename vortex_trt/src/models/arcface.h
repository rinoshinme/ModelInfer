#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "core/logger.h"


namespace vortextrt
{
    class Arcface
    {
    private:
        Logger m_Logger;
        nvinfer1::IEngine* m_Engine;
        nvinfer1::IExecutionContext* m_Context;
        uint32_t m_InputWidth;
        uint32_t m_InputHeight;
        uint32_t m_InputChannels;

        std::vector<float> m_InputBuffer;
        std::vector<float> m_OutputBuffer;
        float* m_InputBufferDevice;
        float* m_OutputBufferDevice;
        cudaStream_t m_Stream;  // cuda stream for synchronization.

    public:
        Arcface(const std::string& model_path);
        ~Arcface();

        void Infer(const cv::Mat& image);

    private:
        bool LoadFile(const std::string& file_path, std::vector<unsigned char>& data);

    };
}
