#include "kernel_operator.h"

using namespace AscendC;


class KernelReplicationPad2d {
public:
    __aicore__ inline KernelReplicationPad2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t size,uint32_t total_size, uint32_t depth, uint32_t height, uint32_t width, uint32_t p_array[])
    {
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x , size);//初始元素个数
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y , total_size);//填充后元素个数

        this->depth = depth;
        this->height = height;
        this->width = width;
        this->pad_top = p_array[2];
        this->pad_left = p_array[0];

        this->output_height = height + p_array[2] + p_array[3];
        this->output_width = width + p_array[0] + p_array[1];
        this->output_hw = this->output_height * this->output_width;
        this->hw = this->height * this->width;
    }

    __aicore__ inline void Process()
    {
        
        for(int d = 0; d < this->depth; d++){
            for (int h = 0; h < this->output_height; h++) {
                for (int w = 0; w < this->output_width; w++) {
                    int input_h = h - this->pad_top;
                    int input_w = w - this->pad_left;
                    // Clamp the input indices to the valid range
                    input_h = (input_h < 0) ? 0 : (input_h >= this->height ? this->height - 1 : input_h);
                    input_w = (input_w < 0) ? 0 : (input_w >= this->width ? this->width - 1 : input_w);

                    // Calculate the output index
                    int output_index = d * this->output_hw + h * this->output_width + w;

                    // Calculate the input index
                    int input_index = d * this->hw + input_h * this->width + input_w;

                    // Copy the value from input to output
                    // output[output_index] = input[input_index];
                    yGm.SetValue(output_index,xGm.GetValue(input_index));

                    // printf(":%d?%d", output_index, input_index);

                }
            }
        }
    }

private:

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    int32_t depth;
    int32_t height;
    int32_t width;
    int32_t pad_top;
    int32_t pad_left;
    int32_t output_height;
    int32_t output_width;
    int32_t output_hw;
    int32_t hw;

};

extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelReplicationPad2d op;
    op.Init(x, y, tiling_data.size, tiling_data.total_size, tiling_data.depth, tiling_data.height, tiling_data.width, tiling_data.p_array);
    op.Process();
}
