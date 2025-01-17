
#include "replication_pad2d_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ReplicationPad2dTilingData tiling;
    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    auto tensor_p = context->GetInputTensor(1);

    uint32_t p_array[4];//左右上下
    for(int i = 0; i < 4; i++){p_array[i] = tensor_p->GetData<int32_t>()[i];}
    
    int32_t x_dimensional = shape_x.GetDimNum();
    int32_t depth = 1;
    for(int i = 0; i < x_dimensional - 2; i++)
    {depth *= shape_x.GetDim(i);}

    int height = shape_x.GetDim(x_dimensional - 2);
    int width = shape_x.GetDim(x_dimensional - 1);

    int32_t total_size = depth * (height + p_array[3] + p_array[2]) * (width + p_array[0] + p_array[1]);

    int32_t size = depth * height * width;

    tiling.set_size(size);
    tiling.set_total_size(total_size);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_p_array(p_array);

    // std::cout << "Size: " << size << std::endl;
    // std::cout << "Total Size: " << total_size << std::endl;
    // std::cout << "Depth: " << depth << std::endl;
    // std::cout << "Height: " << height << std::endl;
    // std::cout << "Width: " << width << std::endl;

    // std::cout << "p_array: ";
    // for (int i = 0; i < 4; ++i) {
    //     std::cout << p_array[i] << " ";
    // }

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
class ReplicationPad2d : public OpDef {
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
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

OP_ADD(ReplicationPad2d);
}
