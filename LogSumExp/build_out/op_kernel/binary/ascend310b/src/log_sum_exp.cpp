#include "kernel_operator.h"


using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 
constexpr uint32_t BLOCK_SIZE = 32;

class KernelLogSumExp {
public:
    __aicore__ inline KernelLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t tileSize, uint32_t tailSize, uint32_t tileNum, uint32_t length,uint32_t dimSize, uint32_t dim)
    {


        this->tileSize = tileSize;//最大空间元素个数
        this->tailSize = tailSize; //求和的元素大于ub空间
        this->tileNum = tileNum;//求和的元素均分到ub空间个数
        this->length =  length;//除去该维度的大小
        this->dimSize = dimSize;//求和维度的大小
        this->dim = dim;//求和维度


        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x , blockLength / sizeof(DTYPE_X));
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y , blockLength / sizeof(DTYPE_X));

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(DTYPE_X));
        // pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileSize * sizeof(DTYPE_X));


        pipe.InitBuffer(tmp1, 1 * sizeof(float)); //用于存储Reduce sum result
        pipe.InitBuffer(tmp2, this->tileSize * sizeof(float)); 
        // pipe.InitBuffer(tmp3, this->tileSize * sizeof(float)); 
        // pipe.InitBuffer(tmp4, this->tileSize * sizeof(float)); 
        

        // pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        // pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(half)); 
        // pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));
        // pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {

        for(int32_t i = 0; i < this->length; i++){
            this->processDataNum = this->tileSize;
            for (int32_t j = 0; j < this->tileNum; j++) {

                if (j == this->tileNum - 1) {
                    this->processDataNum = this->tailSize;
                }

                CopyIn(i, j);
                Compute_sum(i, j);
                // CopyOut(i, j);
            }

        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i, int32_t j)
    {

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[i * this->dimSize + j * this->tileSize], (this->processDataNum+31)/32*32);// TODO:目前默认dim = -1
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute_sum(int32_t i, int32_t j)
    {
        if(std::is_same_v<DTYPE_X, float>){
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            auto p1 = tmp1.Get<float>();
            if(j == 0){
                this->sum = 0;
            }

            Exp(xLocal,xLocal,this->processDataNum);
            ReduceSum(p1, xLocal, xLocal, this->processDataNum);

            if(j == this->tileNum - 1){
                Adds(p1, p1, this->sum, 1);
                Ln(p1,p1,1);
                yGm.SetValue(i,p1.GetValue(0));
            }else{
                float var = p1.GetValue(0);
                this->sum += var;
            }
            inQueueX.FreeTensor(xLocal);
        }else{
            AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            if(j == 0){
                this->sum = 0;
            }

            Cast(p2,xLocal,AscendC::RoundMode::CAST_NONE,this->processDataNum);
            Exp(p2,p2,this->processDataNum);
            ReduceSum(p1, p2, p2, this->processDataNum);

            if(j == this->tileNum - 1){
                Adds(p1, p1, this->sum, 1);
                Ln(p1,p1,1);
                yGm.SetValue(i,(half)p1.GetValue(0));
            }else{
                float var = p1.GetValue(0);
                this->sum += var;
            }
            inQueueX.FreeTensor(xLocal);
        }
        
    }

    // __aicore__ inline void Compute_softmax(int32_t i, int32_t j)
    // {
    //     if(std::is_same_v<DTYPE_X, float>){
    //         LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    //         LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            

    //         Duplicate<DTYPE_Y>(yLocal, static_cast<DTYPE_Y>(this->sum), this->processDataNum);
    //         Exp(xLocal,xLocal,this->processDataNum);
    //         Div(yLocal,xLocal,yLocal,this->processDataNum);

    //         inQueueX.FreeTensor(xLocal);
    //         outQueueY.EnQue<DTYPE_Y>(yLocal);
    //     }else{
    //         LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    //         LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

    //         auto p2 = tmp2.Get<float>();
    //         auto p3 = tmp3.Get<float>();
    //         Cast(p2,xLocal,RoundMode::CAST_NONE, this->processDataNum);
    //         Duplicate<float>(p3, static_cast<float>(this->sum), this->processDataNum);
    //         Exp(p2,p2,this->processDataNum);
    //         Div(p3,p2,p3,this->processDataNum);
    //         Cast(yLocal, p3, RoundMode::CAST_RINT,this->processDataNum);
    //         inQueueX.FreeTensor(xLocal);
    //         outQueueY.EnQue<DTYPE_Y>(yLocal);
    //     }
        // outQueueX.EnQue<DTYPE_X>(xxLocal);
    // }

    // __aicore__ inline void CopyOut(int32_t i, int32_t j)
    // {
    //     AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
    //     AscendC::DataCopy(yGm[i * this->dimSize + j * this->tileSize], yLocal, (this->processDataNum+31)/32*32);// TODO:目前默认dim = -1
    //     outQueueY.FreeTensor(yLocal);
    // }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_X> yGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    float sum;
    uint32_t processDataNum;
    uint32_t tileSize; //最大空间元素个数
    uint32_t tailSize; //求和的元素大于ub空间
    uint32_t tileNum; //求和的元素均分到ub空间个数
    uint32_t length;  //除去该维度的大小
    uint32_t dimSize; //求和维度的大小
    uint32_t dim;
};

extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSumExp op;
    op.Init(x, y, tiling_data.blockLength, tiling_data.tileSize, tiling_data.tailSize, tiling_data.tileNum, tiling_data.length,tiling_data.dimSize, tiling_data.dim);
    op.Process();
}

