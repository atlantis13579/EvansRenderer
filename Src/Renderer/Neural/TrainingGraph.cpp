#include "Renderer/Neural/TrainingJob.h"
#include <stdexcept>

namespace Evans::Neural
{
namespace
{
struct Builder;
struct Value { Builder* Graph;std::string Name; };
struct Builder
{
    Json::Array Nodes;
    Value Op(const std::string& op,const std::vector<Value>& values,Json::Object attributes={})
    {
        const std::string name="v"+std::to_string(Nodes.size());Json::Array inputs;
        for(const auto& value:values) inputs.emplace_back(value.Name);
        Nodes.emplace_back(Json::Object{{"op",op},{"inputs",inputs},{"output",name},{"attributes",attributes}});return {this,name};
    }
    Value Ref(const std::string& name) { return {this,name}; }
};
Value operator+(Value a,Value b) { return a.Graph->Op("Add",{a,b}); }
Value operator-(Value a,Value b) { return a.Graph->Op("Sub",{a,b}); }
Value operator*(Value a,Value b) { return a.Graph->Op("Mul",{a,b}); }
Value operator/(Value a,Value b) { return a.Graph->Op("Div",{a,b}); }
Value Scalar(Value a,const std::string& op,float value,bool reverse=false) { return a.Graph->Op("ScalarBinary",{a},{{"op",op},{"value",double(value)},{"reverse",reverse}}); }
Value operator+(Value a,float b) { return Scalar(a,"Add",b); }
Value operator-(Value a,float b) { return Scalar(a,"Sub",b); }
Value operator*(Value a,float b) { return Scalar(a,"Mul",b); }
Value operator-(float a,Value b) { return Scalar(b,"Sub",a,true); }
Value Unary(const std::string& op,Value a) { return a.Graph->Op(op,{a}); }
Value Reshape(Value a,Json::Array shape) { return a.Graph->Op("Reshape",{a},{{"shape",shape}}); }
Value Column(Value a,int index) { return a.Graph->Op("Slice",{a},{{"axis",1},{"start",index},{"end",index+1}}); }
Value Gather(Value a,Value index,int axis=0) { return a.Graph->Op("Gather",{a,index},{{"axis",axis}}); }
Value Clip(Value a,float lo,float hi) { return a.Graph->Op("Clip",{a},{{"min",lo},{"max",hi}}); }
Value Transpose(Value a) { return a.Graph->Op("Transpose",{a},{{"perm",Json::Array{1,0}}}); }
Value Concat(const std::vector<Value>& values,int axis) { return values[0].Graph->Op("Concat",values,{{"axis",axis}}); }
Value DotRow(Value a) { return a.Graph->Op("ReduceSum",{a},{{"axis",1},{"keepdims",true}}); }
Json Spec(Value value,Json::Array shape) { return Json::Object{{"value",value.Name},{"dtype","float32"},{"shape",shape}}; }
}
void WriteTrainingGraph(const JobOptions& options)
{
    Builder b;const int n=options.Points,pixels=options.PixelBatch;
    const auto view=b.Ref("view"),samples=b.Ref("pixels");
    auto means=b.Ref("initial_means")+b.Ref("offset")*0.1f;
    auto scales=Clip(b.Ref("log_scales"),-9,1);
    auto q=b.Ref("rotation");q=q/Unary("Sqrt",DotRow(q*q)+1e-12f);
    auto opacity=b.Ref("opacity"),color=b.Ref("color");
    const auto exported=Concat({means,scales,q,opacity,color},1);
    auto eye=Reshape(Gather(b.Ref("eyes"),view),{1,3});
    auto basis=Reshape(Gather(b.Ref("basis"),view),{3,3});
    auto camera=b.Op("MatMul",{means-eye,basis});
    auto cx=Column(camera,0),cy=Column(camera,1),cz=Column(camera,2),z=Clip(cz,0.05f,10000);
    auto order=b.Op("ArgSort",{Reshape(cz,{n})});
    const float focal=options.Resolution*1.2071067812f;
    auto inverseZ=Scalar(z,"Div",focal,true);
    auto u=cx*inverseZ+options.Resolution*0.5f,v=cy*inverseZ+options.Resolution*0.5f;
    auto w=Column(q,0),x=Column(q,1),y=Column(q,2),zq=Column(q,3);
    const std::array<Value,3> columns={
        Concat({1.f-(y*y+zq*zq)*2.f,(x*y+w*zq)*2.f,(x*zq-w*y)*2.f},1),
        Concat({(x*y-w*zq)*2.f,1.f-(x*x+zq*zq)*2.f,(y*zq+w*x)*2.f},1),
        Concat({(x*zq+w*y)*2.f,(y*zq-w*x)*2.f,1.f-(x*x+y*y)*2.f},1)};
    std::vector<Value> vx,vy;
    for(int i=0;i<3;++i) {
        auto axis=b.Op("MatMul",{columns[i],basis});auto scale=Unary("Exp",Column(scales,i));
        vx.push_back((Column(axis,0)-cx/z*Column(axis,2))*inverseZ*scale);
        vy.push_back((Column(axis,1)-cy/z*Column(axis,2))*inverseZ*scale);
    }
    auto xx=vx[0]*vx[0]+vx[1]*vx[1]+vx[2]*vx[2]+0.3f;
    auto xy=vx[0]*vy[0]+vx[1]*vy[1]+vx[2]*vy[2];
    auto yy=vy[0]*vy[0]+vy[1]*vy[1]+vy[2]*vy[2]+0.3f;
    auto det=Clip(xx*yy-xy*xy,1e-8f,1e20f);
    auto a=Transpose(Gather(yy/det,order)),bb=Transpose(Gather(xy/det*-1.f,order)),c=Transpose(Gather(xx/det,order));
    auto coords=Gather(b.Ref("pixel_coordinates"),samples);
    auto dx=Column(coords,0)-Transpose(Gather(u,order)),dy=Column(coords,1)-Transpose(Gather(v,order));
    auto power=Clip(a*dx*dx+bb*dx*dy*2.f+c*dy*dy,0,1e8f);
    auto visible=b.Op("Cast",{b.Op("Gt",{cz,b.Ref("near")})},{{"to","float32"}});
    auto support=b.Op("Cast",{b.Op("Lt",{power,b.Ref("cutoff")})},{{"to","float32"}});
    auto alpha=Clip(Unary("Exp",power*-0.5f)*Transpose(Gather(Unary("Sigmoid",opacity)*visible,order)),0,0.99f)*support;
    auto logs=Unary("Log",1.f-alpha);
    auto inclusive=b.Op("CumSum",{logs},{{"axis",1}});
    auto weights=Unary("Exp",inclusive-logs)*alpha;
    auto rgb=Gather(Unary("Sigmoid",color),order);
    auto remaining=Unary("Exp",b.Op("ReduceSum",{logs},{{"axis",1},{"keepdims",true}}));
    auto prediction=b.Op("MatMul",{weights,rgb})+remaining*b.Ref("background");
    auto target=Gather(Reshape(Gather(b.Ref("images"),view),{options.Resolution*options.Resolution,3}),samples);
    auto loss=b.Op("MSELoss",{prediction,target});
    // A weak displacement prior stabilizes stochastic silhouette gradients.
    auto regularizer=Unary("MeanAll",b.Ref("offset")*b.Ref("offset"))*0.00001f;
    loss=loss+regularizer;
    Json::Object tensors;
    for(const char* name:{"initial_means","offset","log_scales","rotation","opacity","color","eyes","basis","images","pixel_coordinates"})
        tensors[name]=Json::Object{{"file",std::string(name)+".npy"},{"trainable",std::string(name)=="offset" || std::string(name)=="log_scales" || std::string(name)=="rotation" || std::string(name)=="opacity" || std::string(name)=="color"}};
    tensors["near"]=Json::Object{{"shape",Json::Array{}},{"data",Json::Array{0.05}}};
    tensors["cutoff"]=Json::Object{{"shape",Json::Array{}},{"data",Json::Array{18}}};
    tensors["background"]=Json::Object{{"shape",Json::Array{1,3}},{"data",Json::Array{double(Background[0]),double(Background[1]),double(Background[2])}}};
    const Json graph(Json::Object{{"format","elman.graph"},{"version",1},
        {"inputs",Json::Object{{"view",Json::Object{{"dtype","int32"},{"shape",Json::Array{1}}}},{"pixels",Json::Object{{"dtype","int32"},{"shape",Json::Array{pixels}}}}}},
        {"outputs",Json::Object{{"loss",Spec(loss,{})},{"model",Spec(exported,{n,14})},{"prediction",Spec(prediction,{pixels,3})}}},{"nodes",b.Nodes}});
    const Json config(Json::Object{{"format","elman.training_graph"},{"version",1},{"tensors",tensors},{"graph",graph},
        {"inputs",Json::Object{{"view",Json::Object{{"random_indices",Json::Object{{"count",1},{"upper",options.Views}}}}},
            {"pixels",Json::Object{{"random_indices",Json::Object{{"count",pixels},{"upper",options.Resolution*options.Resolution}}}}}}},
        {"training",Json::Object{{"steps",options.Steps},{"backend",options.Backend},{"seed",42},{"optimizer","adam"},{"learning_rate",0.025},
            {"clip_grad_norm",1.0},{"log_every",50},{"evaluation_batches",16},{"cpu_threads",8},{"require_loss_decrease",true}}},
        {"output",Json::Object{{"metrics",(options.DataDirectory/"training_metrics.json").u8string()},
            {"tensors",Json::Array{Json::Object{{"file",(options.DataDirectory/"model.npy").u8string()},{"values",Json::Object{{"gaussians","model"}}}}}}}}});
    config.write((options.DataDirectory/"training.json").u8string());
}
}
