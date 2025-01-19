#include "kernel_operator.h"


constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t BLOCK_SIZE = 32;

class KernelAsinh {
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalBlock, uint32_t tileNum, uint32_t tileBlock, uint32_t tileLastBlock)
    {
        this->blockLength = totalBlock * BLOCK_SIZE;//总长度（字节数）
        this->tileNum = tileNum;//分割几部分
        this->tileDataNum = tileBlock * BLOCK_SIZE / sizeof(DTYPE_X);//每部分元素数量
        this->tileLastDataNum = tileLastBlock * BLOCK_SIZE / sizeof(DTYPE_X);//最后一部分元素数量
        // this->dataNum = dataNum;
        // 假设 DTYPE_X 是你定义的数据类型

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x , this->blockLength / sizeof(DTYPE_X));
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y , this->blockLength / sizeof(DTYPE_X));

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float)); 
        // pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float)); 
        // pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            // AscendC::printf("[Process] Processing tile %d\n", i);
            if(i == loopCount - 1){
                this->processDataNum = this->tileLastDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE_X, half>){
            auto p1 = tmp1.Get<float>();
            // auto p2 = tmp2.Get<float>();
            auto px = tmp3.Get<float>();
            // auto py = tmp4.Get<float>();
            // AscendC::printf("half\n");
            Cast(px, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            // 计算 x^2
            Mul(p1, px, px, this->processDataNum);
            // 计算 x^2 + 1
            Adds(p1, p1, static_cast<float>(1.0), this->processDataNum);
            // 计算 sqrt(x^2 + 1)
            Sqrt(p1, p1, this->processDataNum);
            // 计算 x + sqrt(x^2 + 1)
            Add(p1, px, p1, this->processDataNum);
            // 计算 ln(x + sqrt(x^2 + 1))
            Ln(px, p1, this->processDataNum);

            Cast(yLocal, px, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            
        }else if constexpr (std::is_same_v<DTYPE_X, float>){
            auto p1 = tmp1.Get<DTYPE_X>();
            // auto p2 = tmp2.Get<DTYPE_X>();
            // AscendC::printf("float\n");
            //负数过大会爆精度
            float var = xLocal.GetValue(0);
            // AscendC::printf("%f", var);
            if(var < 0){
                Mul(p1, xLocal, xLocal, this->processDataNum);
                Adds(p1, p1, static_cast<DTYPE_X>(1.0), this->processDataNum);
                Sqrt(p1, p1, this->processDataNum);
                Sub(p1, p1, xLocal, this->processDataNum);
                Ln(yLocal, p1, this->processDataNum);
                Muls(yLocal, yLocal, static_cast<DTYPE_X>(-1.0), this->processDataNum);
            }else{
                // 计算 x^2
                Mul(p1, xLocal, xLocal, this->processDataNum);
                // 计算 x^2 + 1
                Adds(p1, p1, static_cast<DTYPE_X>(1.0), this->processDataNum);
                // 计算 sqrt(x^2 + 1)
                Sqrt(p1, p1, this->processDataNum);
                // 计算 x + sqrt(x^2 + 1)
                Add(p1, xLocal, p1, this->processDataNum);
                // 计算 ln(x + sqrt(x^2 + 1))
                Ln(yLocal, p1, this->processDataNum);
            }

        }


        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);

    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp3;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tileLastDataNum;
    uint32_t processDataNum;
};


extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinh op;
    op.Init(x, y, tiling_data.totalBlock, tiling_data.tileNum, tiling_data.tileBlock, tiling_data.tileLastBlock);
    op.Process();
}