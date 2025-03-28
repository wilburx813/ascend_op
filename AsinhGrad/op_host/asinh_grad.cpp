
#include "asinh_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr  uint32_t BLOCK_SIZE = 32;
constexpr  uint32_t BUFFER_NUM = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AsinhGradTilingData tiling;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
    // std::cout<<"ubSize: "<<ubSize<<std::endl;

    //获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    uint32_t totalBlock = alignedLength / BLOCK_SIZE;

    uint32_t ubDataNumber = (inputBytes == 2) ? 5 : 3;


    uint32_t ubBlock = (ubSize / BLOCK_SIZE / ubDataNumber) / BUFFER_NUM;
    // std::this_thread::sleep_for(std::chrono::seconds(20));

    uint32_t tileNum = totalBlock / ubBlock;
    uint32_t tileBlock = tileNum == 0 ? totalBlock : ubBlock;
    uint32_t tileLastBlock = tileBlock;
    
    if((totalBlock % ubBlock) != 0){
        tileLastBlock = totalBlock - tileBlock * tileNum;
        tileNum++;
    }
    
    tiling.set_totalBlock(totalBlock * 32 / inputBytes);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLastBlock(tileLastBlock * 32 / inputBytes);
    tiling.set_tileBlock(tileBlock * 32 / inputBytes);

    // 设置块维度
    context->SetBlockDim(1);

    // 将tiling数据保存到缓存区
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
class AsinhGrad : public OpDef {
public:
    explicit AsinhGrad(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(AsinhGrad);
}
