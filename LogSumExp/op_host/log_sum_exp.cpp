
#include "log_sum_exp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;


// namespace optiling {
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
// {

//     LogSumExpTilingData tiling;

//     uint64_t ubSize;
//     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
//     // auto socVersion = ascendcPlatform.GetSocVersion();
//     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

//     uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
//     uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
//     uint32_t inputLength = inputBytes * inputNum; //输入长度
//     uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
//     uint32_t totalBlock = alignedLength / BLOCK_SIZE;

//     uint32_t ubDataNumber = 2;
//     // uint32_t ubDataNumber = (inputBytes == 2) ? 6 : 4;
//     uint32_t ubBlock = (ubSize / BLOCK_SIZE / ubDataNumber) / BUFFER_NUM;
    

//     uint32_t tileNum = totalBlock / ubBlock;
//     uint32_t tileBlock = tileNum == 0 ? totalBlock : ubBlock;
//     uint32_t tileLastBlock = tileBlock;
    
//     if((totalBlock % ubBlock) != 0){
//         tileLastBlock = totalBlock - tileBlock * tileNum;
//         tileNum++;
//     }
//     auto shape_x = context->GetInputTensor(0)->GetOriginShape();
//     auto attr = context->GetAttrs()->GetListInt({0});
//     uint32_t num_dims = attr->GetSize();
//     uint32_t ndim = shape_x.GetDimNum();
//     uint32_t dims[50];
//     for(int i = 0; i < num_dims; i++){
//         int temp = attr->GetData()[i];
//         if(temp < 0) temp += ndim;
//         dims[i] = temp;
//     }
    
//     uint32_t shape[50];
//     for(int i = 0; i < ndim; i++){
//         shape[i] =  shape_x.GetDim(i);
//     }
//     uint32_t output_size = 1;
//     for (int i = 0; i < ndim; i++) {
//         int is_reduced = 0;
//         for (int j = 0; j < num_dims; j++) {
//             if (i == dims[j]) {
//                 is_reduced = 1;
//                 break;
//             }
//         }
//         if (!is_reduced) {
//             output_size *= shape[i];
//         }
//     }

//     uint32_t dimSize;   
//     for(int i = 0; i < ndim; i++)
//     {
//         if(i == dims[0]){
//             dimSize = shape_x.GetDim(i);
//             continue;
//         }
//         // length *= shape_x.GetDim(i);
//     }


//     tiling.set_tileNum(tileNum);
//     tiling.set_tileDataNum(tileBlock * BLOCK_SIZE / inputBytes);
//     tiling.set_tileLastDataNum(tileLastBlock * BLOCK_SIZE / inputBytes);
//     tiling.set_inputBytes(inputBytes);
//     tiling.set_input_size(inputNum); //输入元素大小
//     tiling.set_output_size(output_size);
//     tiling.set_num_dims(num_dims); //规约维度数量
//     tiling.set_dims(dims); //规约维度索引
//     tiling.set_ndim(ndim); //输入数组的维度数量
//     tiling.set_shape(shape); //输入数组形状


//     std::cout << "tileNum: " << tileNum << std::endl;
//     std::cout << "tileBlock: " << tileBlock << std::endl;
//     std::cout << "tileLastBlock: " << tileLastBlock << std::endl;
//     std::cout << "inputBytes: " << inputBytes << std::endl;
//     std::cout << "inputNum: " << inputNum << std::endl;
//     std::cout << "output_size: " << output_size << std::endl;
//     std::cout << "num_dims: " << num_dims << std::endl;

//     // 打印 dims 数组
//     std::cout << "dims: ";
//     for (int i = 0; i < num_dims; ++i) {
//         std::cout << dims[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "ndim: " << ndim << std::endl;

//     // 打印 shape 数组
//     std::cout << "shape: ";
//     for (int i = 0; i < ndim; ++i) {
//         std::cout << shape[i] << " ";
//     }

//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

//   return ge::GRAPH_SUCCESS;
// }
// }
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    LogSumExpTilingData tiling;
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度
    uint32_t alignedLength = (inputLength + BLOCK_SIZE - 1) & ~ (BLOCK_SIZE - 1);
    tiling.set_blockLength(alignedLength);

    // int32_t dim = *context->GetAttrs()->GetInt(0);
    auto attr = context->GetAttrs()->GetListInt({0});
    auto dim = attr->GetData()[0];
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
class LogSumExp : public OpDef {
public:
    explicit LogSumExp(const char* name) : OpDef(name)
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
        this->Attr("dim").AttrType(OPTIONAL).ListInt(std::vector<int64_t>{0});
        this->Attr("keep_dim").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(LogSumExp);
}
