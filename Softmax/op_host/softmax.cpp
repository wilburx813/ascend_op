
#include "softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftmaxTilingData tiling;
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    tiling.set_blockLength(alignedLength);

    int32_t dim = *context->GetAttrs()->GetInt(0);
    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    int32_t x_dimensional = shape_x.GetDimNum();
    if(dim < 0){
        // int32_t x_dimensional = context->GetInputTensor(0)->GetOriginShape().GetDimNum();
        dim += x_dimensional;
    }
    uint32_t length = 1; 
    uint32_t dimSize;   
    for(int i = 0; i < x_dimensional; i++)
    {
        if(i == dim){
            dimSize = shape_x.GetDim(i);
            continue;
        }
        length *= shape_x.GetDim(i);
    }

    tiling.set_length(length);
    tiling.set_dimSize(dimSize);
    tiling.set_dim(dim);

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
    
    int32_t ub_size = ubSize / 5 / 3 / 32 * 32;

    int32_t dim_length = dimSize * inputBytes; 
    uint32_t tileSize, tailSize, tileNum;
    if(dim_length < ub_size){
        tileSize = dim_length / inputBytes;
        tailSize = tileSize;
        tileNum = 1;
    }else{
        tileSize = ub_size / inputBytes;
        tileNum =  dim_length / ub_size;
        tailSize = dim_length - ub_size * tileNum;
        tileNum++;
    }
    
    tiling.set_tileSize(tileSize);
    tiling.set_tailSize(tailSize);
    tiling.set_tileNum(tileNum);

    std::cout << "inputNum: " << inputNum << std::endl;
    std::cout << "inputBytes: " << inputBytes << std::endl;
    std::cout << "inputLength: " << inputLength << std::endl;
    std::cout << "alignedLength: " << alignedLength << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "x_dimensional: " << x_dimensional << std::endl;
    std::cout << "length: " << length << std::endl;
    std::cout << "dimSize: " << dimSize << std::endl;
    std::cout << "ubSize: " << ubSize << std::endl;
    std::cout << "ub_size: " << ub_size << std::endl;
    std::cout << "dim_length: " << dim_length << std::endl;
    std::cout << "tileSize: " << tileSize << std::endl;
    std::cout << "tailSize: " << tailSize << std::endl;
    std::cout << "tileNum: " << tileNum << std::endl;
    
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
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
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
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Softmax);
}
