#pragma once

#include <string>
#include <memory>
#include "NvInfer.h"
#include "core/logger.h"


namespace vortex
{  
    /* 
        base class for engine inference 
        support batching
    */
    class SimpleInferEngine
    {
    protected:
        Logger m_Logger;
        nvinfer1::ICudaEngine* m_Engine = nullptr;
        nvinfer1::IExecutionContext* m_Context = nullptr;
        nvinfer1::IRuntime* m_Runtime = nullptr;

    public:
        virtual ~SimpleInferEngine();

        virtual bool LoadEngine(const std::string& engine_path);
        virtual void Infer(cv::Mat& image, std::vector<float>& output) = 0;

    private:
        bool LoadEngineData(const std::string& file_path, std::vector<unsigned char>& data);
    };

    // std::shared_ptr<SimpleInferEngine> MakeInferEngine();
}
