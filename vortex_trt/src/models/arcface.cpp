#include "arcface.h"
#include <fstream>
#include "core/cuda_utils.h"


namespace vortex
{
    Arcface::Arcface(const std::string& model_path)
    {
        // load data from file
        std::vector<unsigned char> engine_data;
        bool ret = LoadFile(model_path, engine_data);
        assert(ret);

        // create engine
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(m_Logger);
        assert(runtime != nullptr);
        m_Engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        assert(m_Engine != nullptr);
        m_Context = m_Engine->createExecutionContext();
        assert(m_Context != nullptr);

        // create cuda stream
        checkRuntime(cudaStreamCreate(m_Stream));

        // allocate buffer
        m_InputWidth = 112;
        m_InputHeight = 112;
        m_InputChannels = 3;
        uint32_t input_numel = m_InputWidth * m_InputHeight * m_InputChannels;
        uint32_t output_numel = 512;
        m_InputBuffer.resize(input_numel);
        m_OutputBuffer.resize(output_numel);
        
        checkRuntime(cudaMalloc(m_InputBufferDevice, input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(m_OutputBufferDevice, output_numel * sizeof(float)));
    }

    Arcface::~Arcface()
    {
        // destroy engine and execution context
    }

    void Arcface::Infer(const cv::Mat& image)
    {
        // preprocess
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(112, 112));
        cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);

        uint32_t stride = m_InputWidth * m_InputHeight;
        float* pBlue = &m_InputBuffer[0];
        float* pGreen = &m_InputBuffer[stride];
        float* pRed = &m_InputBUffer[stride * 2];
        unsigned char* pImage = temp.data;
        for (int i = 0; i < stride * 3; i += 3)
        {
            pRed[i] = (pImage[i + 0] / 255.0 - 0.5) / 0.5;
            pGreen[i] = (pImage[i + 1] / 255.0 - 0.5) / 0.5;
            pBlue[i] = (pImage[i + 2] / 255.0 - 0.5) / 0.5;
        }

        // infer
        int input_batch = 1;
        uint32_t input_numel = m_InputWidth * m_InputHeight * m_InputChannels;
        uint32_t output_numel = 512;

        checkRuntime(cudaMemcpyAsync(m_InputBufferDevice, m_InputBuffer.data(), input_numel * sizeof(float), cudaMemcpyHostToDevice, m_Stream));
        auto input_dims = m_Engine->getBindingDimensions(0);
        input_dims.d[0] = input_batch;

    }

    bool Arcface::LoadFile(const std::string& file_path, std::vector<unsigned char>& data)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size_t file_size = file.tellg();
            file.seekg(0, file.beg);
            data.resize(file_size);
            file.read(&data[0], file_size);
            file.close();
            return true;
        }
        return false;
    }
    
}
