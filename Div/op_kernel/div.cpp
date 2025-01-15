#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t BLOCK_SIZE = 32;

class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalBlock, uint32_t tileNum, uint32_t tileBlock, uint32_t tileLastBlock)
    {
        this->blockLength = totalBlock * BLOCK_SIZE;//总长度（字节数）
        this->tileNum = tileNum;//分割几部分
        this->tileDataNum = tileBlock * BLOCK_SIZE / sizeof(DTYPE_X1);//每部分元素数量
        this->tileLastDataNum = tileLastBlock * BLOCK_SIZE / sizeof(DTYPE_X1);//最后一部分元素数量

        // AscendC::printf("[Init] blockLength = %u, tileNum = %u, tileLength = %u,tileLastLength = %u\n", blockLength, tileNum, tileLength,tileLastLength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 , this->blockLength / sizeof(DTYPE_X1));
        yGm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 , this->blockLength / sizeof(DTYPE_X1));
        zGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y , this->blockLength / sizeof(DTYPE_X1));

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));

        // Datatype type = half;

        if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half)); // int8 ->half
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        }else if constexpr(std::is_same_v<DTYPE_X1, int32_t>){
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float)); // int32 ->float32
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        }
        // 打印调试信息
        // AscendC::printf("[Init] blockLength = %u, tileNum = %u, tileLength = %u\n", blockLength, tileNum, tileLength);
        // AscendC::printf("[Init] BlockIdx = %u, BlockNum = %u\n", AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        // AscendC::printf("[Process] loopCount = %d\n", loopCount);

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
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();

        AscendC::DataCopy(x1Local, xGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, yGm[progress * this->tileDataNum], this->processDataNum);

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);

        // 打印输入张量内容
        // AscendC::printf("[CopyIn] progress = %d, tileLength = %u\n", progress, tileLength);

    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(p2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Div(p2, p1, p2, this->processDataNum);
            AscendC::Cast(yLocal, p2, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        }else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(p2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Div(p2, p1, p2, this->processDataNum);
            AscendC::Cast(yLocal, p2, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);

            // AscendC::DumpTensor(p2,94,this->processDataNum); // 使用正确的 DumpTensor 函数
            // AscendC::DumpTensor(yLocal,95,this->processDataNum); // 使用正确的 DumpTensor 函数
        }else{
            AscendC::Div(yLocal, x1Local, x2Local, this->processDataNum);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);

        // 打印输出张量内容
    //     AscendC::printf("[Compute] progress = %d, tileLength = %u\n", progress, tileLength);
    //     AscendC::printf("x1Local:\n");
    //     AscendC::DumpTensor(x1Local,72,tileLength); // 使用正确的 DumpTensor 函数
    //     AscendC::printf("x2Local:\n");
    //     AscendC::DumpTensor(x2Local,74,tileLength); // 使用正确的 DumpTensor 函数
    //     AscendC::printf("yLocal:\n");
    //     AscendC::DumpTensor(yLocal,76,tileLength); // 使用正确的 DumpTensor 函数
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<DTYPE_X1> xGm;
    AscendC::GlobalTensor<DTYPE_X2> yGm;
    AscendC::GlobalTensor<DTYPE_Y> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tileLastDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDiv op;
    op.Init(x1, x2, y, tiling_data.totalBlock, tiling_data.tileNum, tiling_data.tileBlock, tiling_data.tileLastBlock);
    op.Process();
}