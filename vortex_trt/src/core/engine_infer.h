#pragma once


namespace vortex
{  
    /* base class for engine inference */
    class EngineInfer
    {
    private:
    
    public:
        virtual void Infer() = 0;
    };
}
