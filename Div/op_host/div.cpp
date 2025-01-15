
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
    // std::cout<<"ubSize: "<<ubSize<<std::endl;

    //获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    uint32_t totalBlock = alignedLength / BLOCK_SIZE;

    uint32_t ubDataNumber;
    auto type = context->GetInputDesc(0)->GetDataType();
    if(inputBytes == 1){//int8
        ubDataNumber = 7;
    }else if(type == 3 || type == 5){ //int32
        ubDataNumber = 5;
    }else{
        ubDataNumber = 3;
    }
    
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
    // tiling.set_dataType(type);

    // 设置块维度
    context->SetBlockDim(1);

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
