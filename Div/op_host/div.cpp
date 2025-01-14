
#include "div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DivTilingData tiling;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
    std::cout<<"ubSize: "<<ubSize<<std::endl;

    //获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);

    uint32_t ubDataNumber;
    auto type = context->GetInputDesc(0)->GetDataType();
    if(type == 2 || type == 3){//int8 int32
        ubDataNumber = 5;
    }else{
        ubDataNumber = 3;
    }
    
    uint32_t ubBlockNumber = (ubSize / BLOCK_SIZE / ubDataNumber) / BUFFER_NUM;
    uint32_t ubMaxLength = ubBlockNumber * BLOCK_SIZE;

    uint32_t tileNum = alignedLength / ubMaxLength;
    uint32_t tileLength = tileNum == 0 ? alignedLength : ubMaxLength;
    uint32_t tileLastLength = tileLength;

    if((alignedLength % ubMaxLength) != 0){
        tileLastLength = alignedLength - tileLength * tileNum;
        tileNum++;
    }
    

    tiling.set_totalLength(alignedLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLastLength(tileLastLength);
    tiling.set_tileLength(tileLength);
    tiling.set_dataType(type);

    // 设置块维度
    context->SetBlockDim(1);
        

    std::cout << "[TilingFunc] inputNum = " << inputNum << std::endl; // 输出输入数据的元素数量
    std::cout << "[TilingFunc] inputBytes = " << inputBytes << std::endl; // 输出每个元素的大小（字节数）
    std::cout << "[TilingFunc] inputLength = " << inputLength << std::endl; // 输出输入数据的总字节数
    std::cout << "[TilingFunc] alignedLength = " << alignedLength << std::endl; // 输出输入数据的总字节数
    std::cout << "[TilingFunc] ubDataNumber = " << ubDataNumber << std::endl;
    std::cout << "[TilingFunc] ubBlockNumber = " << ubBlockNumber << std::endl;
    std::cout << "[TilingFunc] ubMaxLength = " << ubMaxLength << std::endl;
    std::cout << "[TilingFunc] tileNum = " << tileNum << std::endl;
    std::cout << "[TilingFunc] tileLength = " << tileLength << std::endl;
    std::cout << "[TilingFunc] tileLastLength = " << tileLastLength << std::endl;

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
class Div : public OpDef {
public:
    explicit Div(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Div);
}
