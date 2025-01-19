
#include "asinh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    AsinhTilingData tiling;
    // uint64_t ubSize;
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // auto socVersion = ascendcPlatform.GetSocVersion();
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小


    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    uint32_t totalBlock = alignedLength / BLOCK_SIZE;

    // uint32_t ubDataNumber = 4;
    // uint32_t ubDataNumber = (inputBytes == 2) ? 6 : 4;
    // uint32_t ubBlock = (ubSize / BLOCK_SIZE / ubDataNumber) / BUFFER_NUM;
    uint32_t ubBlock = 992;

    uint32_t tileNum = totalBlock / 992;
    uint32_t tileBlock = 992;
    uint32_t tileLastBlock = tileBlock;
    
    uint32_t temp = totalBlock % 992;
    if((temp) != 0){
        tileLastBlock = temp;
        tileNum++;
    }
    
    tiling.set_totalBlock(totalBlock);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLastBlock(tileLastBlock);
    tiling.set_tileBlock(tileBlock);
    // tiling.set_dataNum(inputNum);
    // tiling.set_dataType(type);
    // std::cout << "ubSize: " << ubSize << std::endl;
    // std::cout << "inputNum: " << inputNum << std::endl;
    // std::cout << "inputBytes: " << inputBytes << std::endl;
    // std::cout << "inputLength: " << inputLength << std::endl;
    // std::cout << "alignedLength: " << alignedLength << std::endl;
    // std::cout << "totalBlock: " << totalBlock << std::endl;
    // std::cout << "ubDataNumber: " << ubDataNumber << std::endl;
    // std::cout << "ubBlock: " << ubBlock << std::endl;
    // std::cout << "tileNum: " << tileNum << std::endl;
    // std::cout << "tileBlock: " << tileBlock << std::endl;
    // std::cout << "tileLastBlock: " << tileLastBlock << std::endl;

    // 设置块维度
    context->SetBlockDim(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
class Asinh : public OpDef {
public:
    explicit Asinh(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
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

OP_ADD(Asinh);
}
