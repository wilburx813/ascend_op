
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsCloseTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileLastBlock);
  TILING_DATA_FIELD_DEF(uint32_t, typekey);
  TILING_DATA_FIELD_DEF(float, rtol);
  TILING_DATA_FIELD_DEF(float, atol);
  TILING_DATA_FIELD_DEF(bool, equal_nan);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsClose, IsCloseTilingData)
}
