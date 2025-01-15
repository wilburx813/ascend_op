
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileLastBlock);
  // TILING_DATA_FIELD_DEF(uint32_t, dataType);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Div, DivTilingData)
}
