
#include "is_close_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

const uint32_t BLOCK_SIZE = 256;
const uint32_t BUFFER_NUM = 2;


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    IsCloseTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    uint32_t totalBlock = alignedLength / BLOCK_SIZE;

    // uint32_t ubDataNumber = 6;
    uint32_t ubDataNumber = (inputBytes == 2) ? 6 : 4;
    uint32_t ubBlock = (ubSize / BLOCK_SIZE / ubDataNumber) / BUFFER_NUM;
    

    uint32_t tileNum = totalBlock / ubBlock;
    uint32_t tileBlock = tileNum == 0 ? totalBlock : ubBlock;
    uint32_t tileLastBlock = tileBlock;
    
    if((totalBlock % ubBlock) != 0){
        tileLastBlock = totalBlock - tileBlock * tileNum;
        tileNum++;
    }
    
    tiling.set_totalBlock(totalBlock);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLastBlock(tileLastBlock);
    tiling.set_tileBlock(tileBlock);

    auto type = context->GetInputDesc(0)->GetDataType();
    float rtol = *context->GetAttrs()->GetFloat(0);
    float atol = *context->GetAttrs()->GetFloat(1);
    bool equal_nan = *context->GetAttrs()->GetBool(2);
    
    tiling.set_typekey(type);
    tiling.set_rtol(rtol);
    tiling.set_atol(atol);
    tiling.set_equal_nan(equal_nan);


    // 设置块维度
    context->SetBlockDim(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class IsClose : public OpDef {
public:
    explicit IsClose(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-05);
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-08);
        this->Attr("equal_nan").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(IsClose);
}
