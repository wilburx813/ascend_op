
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSumExpTilingData)
    // TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    // TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
    // TILING_DATA_FIELD_DEF(uint32_t, tileLastDataNum);
    // TILING_DATA_FIELD_DEF(uint32_t, inputBytes);
    // TILING_DATA_FIELD_DEF(uint32_t, input_size); //输入元素大小
    // TILING_DATA_FIELD_DEF(uint32_t, output_size);
    // TILING_DATA_FIELD_DEF(uint32_t, num_dims); //规约维度数量
    // TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, dims); //规约维度索引
    // TILING_DATA_FIELD_DEF(uint32_t, ndim); //输入数组的维度数量
    // TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, shape) //输入数组形状
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, tailSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, length);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(uint32_t, dim);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSumExp, LogSumExpTilingData)
}
