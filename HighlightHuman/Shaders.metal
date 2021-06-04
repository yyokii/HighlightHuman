/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app's shaders.
*/

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and C code executing Metal API commands. 
#import "ShaderTypes.h"

using namespace metal;

typedef struct {
    float2 position [[attribute(kVertexAttributePosition)]];
    float2 texCoord [[attribute(kVertexAttributeTexcoord)]];
} ImageVertex;


typedef struct {
    float4 position [[position]];
    float2 texCoord;
} ImageColorInOut;

vertex ImageColorInOut capturedImageVertexTransform(ImageVertex in [[stage_in]]) {
    ImageColorInOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

// Convert from YCbCr to rgb
float4 ycbcrToRGBTransform(float4 y, float4 CbCr) {
    const float4x4 ycbcrToRGBTransform = float4x4(
      float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
      float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
      float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
      float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f)
    );

    float4 ycbcr = float4(y.r, CbCr.rg, 1.0);
    return ycbcrToRGBTransform * ycbcr;
}

// This defines the captured image fragment function.
fragment float4 capturedImageFragmentShader(ImageColorInOut in [[stage_in]],
                                            texture2d<float, access::sample> capturedImageTextureY [[ texture(kTextureIndexY) ]],
                                            texture2d<float, access::sample> capturedImageTextureCbCr [[ texture(kTextureIndexCbCr) ]]) {

    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate.
    return ycbcrToRGBTransform(capturedImageTextureY.sample(colorSampler, in.texCoord),
                               capturedImageTextureCbCr.sample(colorSampler, in.texCoord));
}

typedef struct {
    float2 position;
    float2 texCoord;
} CompositeVertex;

typedef struct {
    float4 position [[position]];
    float2 texCoordCamera;
} CompositeColorInOut;

// Composite the image vertex function.
vertex CompositeColorInOut compositeImageVertexTransform(const device CompositeVertex* cameraVertices [[ buffer(0) ]],
                                                         unsigned int vid [[ vertex_id ]]) {
    CompositeColorInOut out;

    const device CompositeVertex& cv = cameraVertices[vid];

    out.position = float4(cv.position, 0.0, 1.0);
    out.texCoordCamera = cv.texCoord;

    return out;
}

// Composite the image fragment function.
fragment half4 compositeImageFragmentShader(CompositeColorInOut in [[ stage_in ]],
                                            texture2d<float, access::sample> capturedImageTextureY [[ texture(0) ]],
                                            texture2d<float, access::sample> capturedImageTextureCbCr [[ texture(1) ]],
                                            texture2d<float, access::sample> whiteColorTexture [[ texture(2) ]],
                                            texture2d<float, access::sample> yellowColorTexture [[ texture(3) ]],
                                            texture2d<float, access::sample> alphaTexture [[ texture(4) ]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 cameraTexCoord = in.texCoordCamera;

    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate.
    float4 rgb = ycbcrToRGBTransform(capturedImageTextureY.sample(s, cameraTexCoord), capturedImageTextureCbCr.sample(s, cameraTexCoord));

    half4 cameraColor = half4(rgb);
    half4 whiteColor = half4(whiteColorTexture.sample(s, cameraTexCoord));
    half4 yellowColor = half4(yellowColorTexture.sample(s, cameraTexCoord)) * 2.0;

    return cameraColor + whiteColor + yellowColor;
}

// （拡大した人体画像　- 人体画像）で人体の縁を作って、それを白、黄のテクスチャとして出力
kernel void matteConvert(texture2d<half, access::read> inTexture [[ texture(0) ]],
                         texture2d<half, access::write> outWhiteTexture [[ texture(1) ]],
                         texture2d<half, access::write> outYellowTexture [[ texture(2) ]],
                         uint2 gid [[thread_position_in_grid]]) {

    uint2 textureIndex(gid);
    if (inTexture.read(textureIndex).r > 0.1) {
        // 人体部分は色なし
        outWhiteTexture.write(half4(0.0), gid);
        outYellowTexture.write(half4(0.0), gid);
        return;
    }

    // 拡大
    constexpr int scale = 15;
    constexpr int radius = scale / 2;
    half color = 0.0;

    for (int i=0; i<scale; i++) {
        for (int j=0; j<scale; j++) {
            uint2 textureIndex(gid.x + (i - radius), gid.y + (j - radius));
            half alpha = inTexture.read(textureIndex).r;
            if (alpha > 0.1) {
                color = 1.0;
                break;
            }
        }
        if (color > 0.0) {
            break;
        }
    }

    outWhiteTexture.write(half4(color, color, color, 1.0), gid);
    outYellowTexture.write(half4(color, color, 0.0, 1.0), gid);
}
