#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t BLOCK_SIZE = 32;

class KernelAsinhGrad {
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint32_t totalBlock, uint32_t tileNum, uint32_t tileBlock, uint32_t tileLastBlock)
    {
        this->blockLength = totalBlock * BLOCK_SIZE;//总长度（字节数）
        this->tileNum = tileNum;//分割几部分
        this->tileDataNum = tileBlock * BLOCK_SIZE / sizeof(DTYPE_Y);//每部分元素数量
        this->tileLastDataNum = tileLastBlock * BLOCK_SIZE / sizeof(DTYPE_Y);//最后一部分元素数量

        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y , this->blockLength / sizeof(DTYPE_Y));
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY *)dy , this->blockLength / sizeof(DTYPE_DY));
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z , this->blockLength / sizeof(DTYPE_Z));

        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_DY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Z));

        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float)); 
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float)); 
        pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));
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
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.AllocTensor<DTYPE_DY>();
        AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);

        inQueueY.EnQue(yLocal);
        inQueueDY.EnQue(dyLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        // output = input_dy / tf.math.cosh(input_y)

        if constexpr (std::is_same_v<DTYPE_Y, half>){
            auto p1 = tmp1.Get<float>();
            // auto p2 = tmp2.Get<float>();
            auto py = tmp2.Get<float>();
            auto pdy = tmp3.Get<float>();
            auto pz = tmp4.Get<float>();
            
            // AscendC::printf("half\n");
            Cast(py, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            Cast(pdy, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            Exp(p1, py, this->processDataNum);
           
            Muls(py, py, static_cast<float>(-1), this->processDataNum);

            Exp(py, py, this->processDataNum);

            Add(p1, py, p1, this->processDataNum);

            Muls(p1, p1, static_cast<float>(0.5), this->processDataNum);

            Div(pz, pdy, p1, this->processDataNum);

            Cast(zLocal, pz, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            
        }else if constexpr (std::is_same_v<DTYPE_Y, float>){
            auto p1 = tmp1.Get<DTYPE_Y>();
            auto p2 = tmp2.Get<DTYPE_Y>();

            Exp(p1, yLocal, this->processDataNum);
           
            Muls(yLocal, yLocal, static_cast<DTYPE_Y>(-1), this->processDataNum);

            Exp(yLocal, yLocal, this->processDataNum);

            Add(yLocal, yLocal, p1, this->processDataNum);

            Muls(yLocal, yLocal, static_cast<DTYPE_Y>(0.5), this->processDataNum);

            Div(zLocal, dyLocal, yLocal, this->processDataNum);
        }

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueDY.FreeTensor(dyLocal);
        inQueueY.FreeTensor(yLocal);

    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp3, tmp4;

    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_DY> dyGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tileLastDataNum;
    uint32_t processDataNum;
};


extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinhGrad op;
    op.Init(y, dy, z, tiling_data.totalBlock, tiling_data.tileNum, tiling_data.tileBlock, tiling_data.tileLastBlock);
    op.Process();
}