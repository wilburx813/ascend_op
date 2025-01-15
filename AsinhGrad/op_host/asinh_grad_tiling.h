
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsinhGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileLastBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsinhGrad, AsinhGradTilingData)
}
