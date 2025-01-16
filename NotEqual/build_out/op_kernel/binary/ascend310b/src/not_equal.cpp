#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t BLOCK_SIZE = 256;

class KernelAsinh {
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalBlock, uint32_t tileNum, uint32_t tileBlock, uint32_t tileLastBlock,uint32_t typekey)
    {
        this->blockLength = totalBlock * BLOCK_SIZE;//总长度（字节数）
        this->tileNum = tileNum;//分割几部分
        this->tileDataNum = tileBlock * BLOCK_SIZE / sizeof(DTYPE_X1);//每部分元素数量
        this->tileLastDataNum = tileLastBlock * BLOCK_SIZE / sizeof(DTYPE_X1);//最后一部分元素数量
        this->typekey = typekey;
        // this->dataNum = dataNum;
        // 假设 DTYPE_X1 是你定义的数据类型

        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 , this->blockLength / sizeof(DTYPE_X1));
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 , this->blockLength / sizeof(DTYPE_X1));
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y , this->blockLength / sizeof(DTYPE_X1));

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));

        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half)); 
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(half)); 
        pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(float));
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
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X1>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {


        if(this->typekey == 2){//int8
            //half
            AscendC::LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
            AscendC::LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> p1 = tmp1.Get<half>();
            AscendC::LocalTensor<half> px1 = tmp2.Get<half>();
            AscendC::LocalTensor<half> px2 = tmp3.Get<half>();

            Cast(px1, x1Local, RoundMode::CAST_NONE, this->processDataNum);
            Cast(px2, x2Local, RoundMode::CAST_NONE, this->processDataNum);

            Duplicate<half>(p1, (half)1.0, this->processDataNum);
            Compare(yLocal, px1, px2, AscendC::CMPMODE::NE, this->processDataNum);//好像要256字节对齐
            Select(p1, yLocal, p1, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<uint8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);

        }else if(this->typekey == 3){//int32
            AscendC::LocalTensor<int32_t> x1Local = inQueueX1.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> x2Local = inQueueX2.DeQue<int32_t>();
            AscendC::LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> p1 = tmp1.Get<half>();
            AscendC::LocalTensor<float> px1 = tmp4.Get<float>();
            AscendC::LocalTensor<float> px2 = tmp5.Get<float>();

            Cast(px1, x1Local, RoundMode::CAST_RINT, this->processDataNum);
            Cast(px2, x2Local, RoundMode::CAST_RINT, this->processDataNum);

            Duplicate<half>(p1, (half)1.0, this->processDataNum);
            Compare(yLocal, px1, px2, AscendC::CMPMODE::NE, this->processDataNum);//好像要256字节对齐
            Select(p1, yLocal, p1, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<uint8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }else if(this->typekey == 1){
            //half
            AscendC::LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            AscendC::LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> p1 = tmp1.Get<half>();

            Duplicate<half>(p1, (half)1.0, this->processDataNum);
            Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::NE, this->processDataNum);//好像要256字节对齐
            Select(p1, yLocal, p1, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueY.EnQue<uint8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }else{
            //float
            AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            AscendC::LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> p1 = tmp1.Get<half>();

            Duplicate<half>(p1, (half)1.0, this->processDataNum);
            Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::NE, this->processDataNum);//好像要256字节对齐
            Select(p1, yLocal, p1, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);//maybe CAST_RINT
            outQueueY.EnQue<uint8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }



    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp3, tmp4, tmp5;
    AscendC::GlobalTensor<DTYPE_X1> x1Gm;
    AscendC::GlobalTensor<DTYPE_X1> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tileLastDataNum;
    uint32_t processDataNum;
    uint32_t typekey;
};


extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinh op;
    op.Init(x1, x2, y, tiling_data.totalBlock, tiling_data.tileNum, tiling_data.tileBlock, tiling_data.tileLastBlock,tiling_data.typekey);
    op.Process();
}
