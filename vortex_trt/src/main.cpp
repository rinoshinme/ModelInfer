#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "models/arcface.h"

using namespace vortex;

int main()
{
    std::string engine_path("../../sample/arcface_resnet50.engine");
    Arcface model(engine_path);
    std::string image_path("../../sample/sample1.jpg");
    cv::Mat image = cv::imread(image_path);

    std::vector<float> outputs;
    model.Infer(image, outputs);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << outputs[i] << std::endl;
    }

    return 0;
}
