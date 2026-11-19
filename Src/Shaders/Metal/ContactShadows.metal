#include <metal_stdlib>
using namespace metal;

// Independent Metal contact visibility. McGuire and Mara, "Efficient GPU
// Screen-Space Ray Tracing" (2014), https://jcgt.org/published/0003/04/04/.
// Traverse finite pixel-cell intervals and intersect a camera-axis depth slab.
// This is supplementary visibility, not a replacement for shadow maps: missing
// screen coverage and exhausted budgets are lit. Each depth sample has its own
// output layer. Material coverage remains pixel-centred, including under MSAA.
// Compile with fast math disabled: compensated products and finite checks below
// are part of the algorithm. No other API's shader is an input to this source.
constant float PixelCenter = 0.5;
constant float ClearDepth = 1.0;
constant float DirectionEpsilon = 1.0e-12;
constant float FloatEpsilon = 1.1920928955078125e-7; // IEEE binary32 spacing at one
constant float NativeRoundoff = 4.0 * FloatEpsilon;
constant float MaximumFiniteDepth = 3.402823466e38;
constant uint ClipPlaneCount = 7;
constant uint MaximumDepthSamples = 8;
constant uint MaximumInstanceCount = 64;

struct TraceConstants
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 DepthMetric; // near, far, perspective flag, positive clip-W guard
    float4 LightPositionType; // raw direction (w=0), or finite centre (w=1)
    float4 LightDirection; // reserved ABI field; preserve raw direction precision
    float4 TraceParameters; // maximum world distance, start bias, axial thickness
    uint4 ImageAndBudget; // width, height, visited-cell budget, sample count
};
struct RasterSamples { float2 Positions[MaximumDepthSamples]; }; // top-left [0,1)

float2 AddWithResidual(float a, float b)
{
    float sum = a + b;
    float recovered = sum - a;
    return float2(sum, (a - (sum - recovered)) + (b - recovered));
}
float2 RowWithResidual(constant float4& row, float4 point)
{
    float4 products = row * point;
    float4 errors = fma(row, point, -products);
    float2 first = AddWithResidual(products.x, products.y);
    float2 second = AddWithResidual(first.x, products.z);
    float2 last = AddWithResidual(second.x, products.w);
    return float2(last.x, first.y + second.y + last.y + dot(errors, float4(1.0)));
}
float3 ReconstructReceiver(float2 raster, float depth, constant TraceConstants& trace)
{
    float2 ndc = raster / float2(trace.ImageAndBudget.xy) * 2.0 - 1.0;
    float4 point(ndc.x, -ndc.y, depth, 1.0);
    float2 denominator = RowWithResidual(trace.InverseViewProjection[3], point);
    float3 world;
    for (uint axis = 0; axis < 3; ++axis)
    {
        float2 numerator = RowWithResidual(trace.InverseViewProjection[axis], point);
        float quotient = numerator.x / denominator.x;
        float residual = fma(-quotient, denominator.x, numerator.x) + numerator.y - quotient * denominator.y;
        world[axis] = quotient + residual / denominator.x;
    }
    return world;
}
float4 ProjectDirection(float3 direction, constant TraceConstants& trace)
{
    float4 result;
    for (uint axis = 0; axis < 4; ++axis)
    {
        float2 value = RowWithResidual(trace.ViewProjection[axis], float4(direction, 0.0));
        result[axis] = value.x + value.y;
    }
    return result;
}
float AxialDepth(float stored, constant TraceConstants& trace)
{
    // Interpolate reciprocals, avoiding near*far overflow for large cameras.
    return trace.DepthMetric.z > PixelCenter ?
        1.0 / ((1.0 - stored) / trace.DepthMetric.x + stored / trace.DepthMetric.y) :
        (1.0 - stored) * trace.DepthMetric.x + stored * trace.DepthMetric.y;
}
float PlaneDistance(float4 point, uint plane, float guard)
{
    switch (plane)
    {
        case 0: return point.w - guard;
        case 1: return point.w + point.x;
        case 2: return point.w - point.x;
        case 3: return point.w + point.y;
        case 4: return point.w - point.y;
        case 5: return point.z;
        default: return point.w - point.z;
    }
}
float ReadDepth(depth2d<float, access::read> depth, uint2 pixel, uint sampleIndex)
{
    (void)sampleIndex;
    return depth.read(pixel);
}
float ReadDepth(depth2d_ms<float, access::read> depth, uint2 pixel, uint sampleIndex)
{
    return depth.read(pixel, sampleIndex);
}
struct ProjectedSegment
{
    float2 FirstPixel, PixelDelta;
    float2 ReciprocalW, DepthOverW;
};
bool ProjectRay(float2 raster, float receiverDepth, constant TraceConstants& trace,
    thread ProjectedSegment& segment)
{
    float3 world = ReconstructReceiver(raster, receiverDepth, trace);
    float3 vector = trace.LightPositionType.w < PixelCenter ? trace.LightPositionType.xyz : trace.LightPositionType.xyz - world;
    if (!all(isfinite(world)) || !all(isfinite(vector))) return false;
    float scale = max(abs(vector.x), max(abs(vector.y), abs(vector.z)));
    if (!(scale > 0.0)) return false;
    float scaledLength = length(vector / scale);
    float3 direction = (vector / scale) / scaledLength;
    float distance = trace.TraceParameters.x;
    // Compare before multiplying: a finite light at 1e20 must not overflow length.
    if (trace.LightPositionType.w > PixelCenter && scale < distance / scaledLength)
        distance = scale * scaledLength;
    float start = trace.TraceParameters.y;
    if (!(distance > start)) return false;

    bool perspective = trace.DepthMetric.z > PixelCenter;
    float receiverW = perspective ? AxialDepth(receiverDepth, trace) : 1.0;
    float2 ndc = raster / float2(trace.ImageAndBudget.xy) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 origin = float4(ndc, receiverDepth, 1.0) * receiverW;
    float4 derivative = ProjectDirection(direction, trace);
    float4 first = fma(derivative, float4(start), origin);
    float4 last = fma(derivative, float4(distance), origin);
    if (!all(isfinite(first)) || !all(isfinite(last))) return false;
    float enter = 0.0, leave = 1.0;
    for (uint plane = 0; plane < ClipPlaneCount; ++plane)
    {
        float a = PlaneDistance(first, plane, trace.DepthMetric.w);
        float b = PlaneDistance(last, plane, trace.DepthMetric.w);
        if (a < 0.0 && b < 0.0) return false;
        if ((a < 0.0) != (b < 0.0))
        {
            float crossing = a / (a - b);
            if (a < 0.0) enter = max(enter, crossing);
            else leave = min(leave, crossing);
        }
        if (!(enter < leave)) return false;
    }
    float2 distanceRange = fma(float2(enter, leave), float2(distance - start), float2(start));
    float2 clipW = fma(float2(derivative.w), distanceRange, float2(receiverW));
    if (!all(clipW > 0.0) || !all(isfinite(clipW))) return false;
    // Anchor traversal to the exact raster sample. Computing two rounded world
    // endpoints and subtracting their projections loses short-ray direction.
    float2 product = ndc * derivative.w;
    float2 residual = fma(ndc, float2(derivative.w), -product);
    float2 slope = ((derivative.xy - product) - residual) *
        float2(trace.ImageAndBudget.xy) * float2(PixelCenter, -PixelCenter);
    segment.FirstPixel = raster + (slope / clipW.x) * distanceRange.x;
    segment.PixelDelta = (slope / clipW.y) * (distanceRange.y - distanceRange.x) * (receiverW / clipW.x);
    segment.ReciprocalW = 1.0 / clipW;
    if (perspective) segment.DepthOverW = float2(1.0); // axial depth equals clip W
    else
    {
        float2 stored = fma(float2(derivative.z), distanceRange, float2(origin.z)) / clipW;
        segment.DepthOverW = float2(AxialDepth(stored.x, trace), AxialDepth(stored.y, trace)) / clipW;
    }
    return all(isfinite(segment.FirstPixel)) && all(isfinite(segment.PixelDelta)) &&
        all(isfinite(segment.DepthOverW));
}
float SegmentDepth(ProjectedSegment segment, float progress)
{
    return mix(segment.DepthOverW.x, segment.DepthOverW.y, progress) /
        mix(segment.ReciprocalW.x, segment.ReciprocalW.y, progress);
}
template<typename DepthTexture>
float ContactVisibility(DepthTexture depth, uint3 receiver, constant TraceConstants& trace,
    constant RasterSamples& samples)
{
    float stored = ReadDepth(depth, receiver.xy, receiver.z);
    if (!(stored >= 0.0 && stored < ClearDepth)) return 1.0;
    ProjectedSegment segment;
    if (!ProjectRay(float2(receiver.xy) + samples.Positions[receiver.z], stored, trace, segment)) return 1.0;
    float major = max(abs(segment.PixelDelta.x), abs(segment.PixelDelta.y));
    float minor = min(abs(segment.PixelDelta.x), abs(segment.PixelDelta.y));
    if (major <= DirectionEpsilon) return 1.0;
    float pixelUncertainty = NativeRoundoff * float(max(trace.ImageAndBudget.x, trace.ImageAndBudget.y));
    float intervalTolerance = pixelUncertainty / major;
    float cornerTolerance = intervalTolerance + (minor > DirectionEpsilon ? pixelUncertainty / minor : 0.0);
    int2 extent(trace.ImageAndBudget.xy), cell, advance;
    float2 nextCrossing, crossingStep;
    for (uint axis = 0; axis < 2; ++axis)
    {
        float origin = segment.FirstPixel[axis], delta = segment.PixelDelta[axis];
        float lower = floor(origin);
        cell[axis] = clamp(int(lower) - (delta < 0.0 && origin == lower ? 1 : 0), 0, extent[axis] - 1);
        advance[axis] = delta < 0.0 ? -1 : 1;
        nextCrossing[axis] = crossingStep[axis] = INFINITY;
        if (abs(delta) > DirectionEpsilon)
        {
            float boundary = float(cell[axis] + (advance[axis] > 0 ? 1 : 0));
            nextCrossing[axis] = max(0.0, (boundary - origin) / delta);
            crossingStep[axis] = 1.0 / abs(delta);
        }
    }
    float begin = 0.0;
    for (uint visited = 0; visited < trace.ImageAndBudget.z && begin < 1.0; ++visited)
    {
        if (any(cell < 0) || any(cell >= extent)) break;
        float finish = min(1.0, min(nextCrossing.x, nextCrossing.y));
        if (finish > begin + intervalTolerance && any(cell != int2(receiver.xy)))
        {
            float sampleDepth = ReadDepth(depth, uint2(cell), receiver.z);
            if (sampleDepth >= 0.0 && sampleDepth < ClearDepth)
            {
                float front = AxialDepth(sampleDepth, trace);
                float back = front + min(trace.TraceParameters.z, MaximumFiniteDepth - front);
                float a = SegmentDepth(segment, begin), b = SegmentDepth(segment, finish);
                float uncertainty = NativeRoundoff * max(1.0, max(max(abs(front), abs(back)), max(abs(a), abs(b))));
                // A continuous projective interval intersects this finite slab.
                // Tangency is unknown; a real zero-thickness crossing still hits.
                if (max(a, b) - front >= uncertainty && back - min(a, b) >= uncertainty) return 0.0;
            }
        }
        if (finish >= 1.0) break;
        begin = max(begin, finish);
        for (uint axis = 0; axis < 2; ++axis)
            if (nextCrossing[axis] <= finish + cornerTolerance)
            {
                // Both corner edges must be crossed before entering the diagonal
                // cell. Nearly parallel rays must never rewind interval progress.
                begin = max(begin, nextCrossing[axis]);
                cell[axis] += advance[axis];
                nextCrossing[axis] += crossingStep[axis];
            }
    }
    return 1.0;
}
kernel void ContactVisibilitySingle(depth2d<float, access::read> depth [[texture(0)]],
    texture2d_array<float, access::write> result [[texture(1)]], constant TraceConstants& trace [[buffer(0)]],
    constant RasterSamples& samples [[buffer(1)]], uint3 pixel [[thread_position_in_grid]])
{
    if (any(pixel.xy >= trace.ImageAndBudget.xy) || pixel.z >= trace.ImageAndBudget.w) return;
    result.write(float4(ContactVisibility(depth, pixel, trace, samples)), pixel.xy, pixel.z);
}
kernel void ContactVisibilityMultisample(depth2d_ms<float, access::read> depth [[texture(0)]],
    texture2d_array<float, access::write> result [[texture(1)]], constant TraceConstants& trace [[buffer(0)]],
    constant RasterSamples& samples [[buffer(1)]], uint3 pixel [[thread_position_in_grid]])
{
    if (any(pixel.xy >= trace.ImageAndBudget.xy) || pixel.z >= trace.ImageAndBudget.w) return;
    result.write(float4(ContactVisibility(depth, pixel, trace, samples)), pixel.xy, pixel.z);
}

// The optional Forward prerequisite shares material coverage, not light-space
// shadow bias or stochastic translucent casting. No material MRT/HDR is needed.
struct ContactVertex { packed_float3 Position; packed_float3 Normal; packed_float4 Color; packed_float2 UV; };
struct ContactDraw
{
    float4 World[4], NormalWorld[4], ViewProjection[4], LightViewProjection[4];
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
};
struct ContactObject { float4 World[4], NormalWorld[4], PreviousWorldViewProjection[4]; float4 Wind, Gust, VegetationHistory; };
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
float3 VegetationOffset(float3 world, float height, float4 wind, float4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return float3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
float4 VegetationPoint(float3 position,float2 uv,constant ContactObject& data)
{
    if (data.Gust.w<0.5) return float4(position,1);
    float4 local=float4(position,1);
    float3 world=float3(dot(data.World[0],local),dot(data.World[1],local),dot(data.World[2],local));
    float height=max(0.0f,data.Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z);
    return float4(position+data.NormalWorld[0].xyz*offset.x+data.NormalWorld[1].xyz*offset.y+data.NormalWorld[2].xyz*offset.z,1);
}
struct ContactInstances
{
    float4 ViewProjection[4];
    float4 Material;
    ContactObject Objects[MaximumInstanceCount];
};
struct ContactCoverage
{
    float4 Position [[position]];
    float2 UV [[center_perspective]];
    float Alpha [[center_perspective]];
};
float4 TransformContact(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}
ContactCoverage DepthCoverage(device const ContactVertex& input, constant float4* world, constant float4* camera)
{
    ContactCoverage output;
    output.Position = TransformContact(camera, TransformContact(world, float4(float3(input.Position), 1.0)));
    output.UV = float2(input.UV.x, 1.0 - input.UV.y);
    output.Alpha = input.Color.w;
    return output;
}
vertex ContactCoverage ContactDepthVertex(device const ContactVertex* vertices [[buffer(0)]],
    constant ContactDraw& draw [[buffer(1)]], uint vertexId [[vertex_id]])
{
    return DepthCoverage(vertices[vertexId], draw.World, draw.ViewProjection);
}
vertex ContactCoverage ContactDepthInstanceVertex(device const ContactVertex* vertices [[buffer(0)]],
    constant ContactInstances& instances [[buffer(3)]], uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    ContactCoverage output=DepthCoverage(vertices[vertexId], instances.Objects[instanceId].World, instances.ViewProjection);
    output.Position=TransformContact(instances.ViewProjection,TransformContact(instances.Objects[instanceId].World,
        VegetationPoint(float3(vertices[vertexId].Position),float2(vertices[vertexId].UV),instances.Objects[instanceId])));
    return output;
}
fragment void ContactDepthFragment(ContactCoverage input [[stage_in]], constant ContactDraw& draw [[buffer(1)]],
    texture2d<float> diffuse [[texture(1)]], texture2d<float> opacity [[texture(3)]], sampler materialSampler [[sampler(0)]])
{
    float coverage = input.Alpha * draw.MaterialParameters.y;
    if (draw.TextureFlags.x != 0u) coverage *= diffuse.sample(materialSampler, input.UV).a;
    if (draw.TextureFlags.z != 0u) coverage *= opacity.sample(materialSampler, input.UV).r;
    if (saturate(coverage) < draw.MaterialParameters.z) discard_fragment();
}
