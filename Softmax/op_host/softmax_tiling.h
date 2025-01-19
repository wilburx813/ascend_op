
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, tailSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, length);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}
