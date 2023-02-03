#pragma once


namespace vortex
{  
    /* 
        base class for engine inference 
        support batching
    */
    class EngineInfer
    {
    private:
    
    public:

        virtual void Preprocess();
        virtual void Infer() = 0;
    };
}
