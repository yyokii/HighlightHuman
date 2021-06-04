/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The host app renderer.
*/

import Foundation
import Metal
import MetalPerformanceShaders
import MetalKit
import ARKit

let kMaxBuffersInFlight: Int = 3

let kImagePlaneVertexData: [Float] = [
    -1.0, -1.0, 0.0, 1.0,
    1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0
]

class Renderer {
    let session: ARSession
    let matteGenerator: ARMatteGenerator
    let device: MTLDevice
    let inFlightSemaphore = DispatchSemaphore(value: kMaxBuffersInFlight)
    var mtkView: MTKView

    var commandQueue: MTLCommandQueue!
    var imagePlaneVertexBuffer: MTLBuffer!
    // 最終画像合成用PipelineState
    var compositePipelineState: MTLRenderPipelineState!
    // 人体画像の拡大加工用PipelineState
    var computeState: MTLComputePipelineState!
    // キャプチャ画像テクスチャ
    var capturedImageTextureY: CVMetalTexture?
    var capturedImageTextureCbCr: CVMetalTexture?
    var capturedImageTextureCache: CVMetalTextureCache!
    // 人体画像テクスチャ
    var alphaTexture: MTLTexture?       // 人体画像
    var whiteBlurTexture: MTLTexture!   // 人体画像を白色にして拡大・ブラーしたテクスチャ
    var yellowBlurTexture: MTLTexture!  // 人体画像を黄色にして拡大・ブラーしたテクスチャ
    // 画面サイズ
    var viewportSize: CGSize = CGSize()
    var viewportSizeDidChange: Bool = false
    // 人体画像加工時のコンピュートシェーダーのスレッドグループサイズ
    var threadgroupSize = MTLSizeMake(32, 32, 1)
    // アニメーションカウント
    var time = 0

    init(session: ARSession, metalDevice device: MTLDevice, mtkView: MTKView) {
        self.session = session
        self.device = device
        self.mtkView = mtkView
        matteGenerator = ARMatteGenerator(device: device, matteResolution: .half)
        loadMetal()
    }

    func drawRectResized(size: CGSize) {
        viewportSize = size
        viewportSizeDidChange = true
    }

    func update() {
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        // レンダリング中にカメラキャプチャしたテクスチャが解放されないように保持
        var textures = [capturedImageTextureY, capturedImageTextureCbCr]
        commandBuffer.addCompletedHandler { [weak self] commandBuffer in
            if let strongSelf = self {
                strongSelf.inFlightSemaphore.signal()
            }
            textures.removeAll()
        }
        // カメラキャプチャテクスチャ取得（Y、CbCrの２つ)
        guard let currentFrame = session.currentFrame else { return }
        let pixelBuffer = currentFrame.capturedImage
        if CVPixelBufferGetPlaneCount(pixelBuffer) < 2 { return }
        capturedImageTextureY = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
        // 画面サイズに応じてuv座標を設定
        if viewportSizeDidChange {
            viewportSizeDidChange = false
            // Update the texture coordinates of our image plane to aspect fill the viewport
            let displayToCameraTransform = currentFrame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()

            let vertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
            for index in 0...3 {
                let textureCoordIndex = 4 * index + 2   // kImagePlaneVertexData が 頂点座標(x,y) + uv座標(u,v)になっている。uv設定するので +2
                let textureCoord = CGPoint(x: CGFloat(kImagePlaneVertexData[textureCoordIndex]), y: CGFloat(kImagePlaneVertexData[textureCoordIndex + 1]))
                let transformedCoord = textureCoord.applying(displayToCameraTransform)
                // キャプチャ画像
                vertexData[textureCoordIndex] = Float(transformedCoord.x)
                vertexData[textureCoordIndex + 1] = Float(transformedCoord.y)
            }
        }
        // 人体画像取得
        alphaTexture = matteGenerator.generateMatte(from: currentFrame, commandBuffer: commandBuffer)
        // ブラーの効果を人体より大きく見せたいので人体画像を拡大。ついでに白色、黄色の２色分のテクスチャを生成。
        if let width = alphaTexture?.width, let height = alphaTexture?.height {
            let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                     width: width, height: height, mipmapped: false)
            colorDesc.usage = [.shaderRead, .shaderWrite]
            whiteBlurTexture = device.makeTexture(descriptor: colorDesc)
            yellowBlurTexture = device.makeTexture(descriptor: colorDesc)

            let threadCountW = (width + self.threadgroupSize.width - 1) / self.threadgroupSize.width
            let threadCountH = (height + self.threadgroupSize.height - 1) / self.threadgroupSize.height
            let threadgroupCount = MTLSizeMake(threadCountW, threadCountH, 1)

            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

            computeEncoder.setComputePipelineState(computeState)
            computeEncoder.setTexture(alphaTexture, index: 0)
            computeEncoder.setTexture(whiteBlurTexture, index: 1)
            computeEncoder.setTexture(yellowBlurTexture, index: 2)
            computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            computeEncoder.endEncoding()
        }
        // 時間でブラーの大きさを変える
        time += 1
        // ブラー（白）
        let whiteIntensity = Int((sin(Float(time)/3) + 2) * 30) | 0x01  // MPSImageTentのサイズには奇数を指定する必要がある。
        let kernel1 = MPSImageTent(device: device, kernelWidth: whiteIntensity, kernelHeight: whiteIntensity)
        kernel1.encode(commandBuffer: commandBuffer,
                      inPlaceTexture: &whiteBlurTexture!, fallbackCopyAllocator: nil)
        // ブラー（黄）
        let yellowIntensity = Int((sin(Float(time)/3) + 2) * 100) | 0x01
        let kernel2 = MPSImageTent(device: device, kernelWidth: yellowIntensity, kernelHeight: yellowIntensity)
        kernel2.encode(commandBuffer: commandBuffer,
                      inPlaceTexture: &yellowBlurTexture!, fallbackCopyAllocator: nil)
        // キャプチャ画像＋ブラー（白・黄色）合成
        guard let renderPassDescriptor = mtkView.currentRenderPassDescriptor, let currentDrawable = mtkView.currentDrawable else { return }
        let compositeRenderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        compositeImagesWithEncoder(renderEncoder: compositeRenderEncoder)
        compositeRenderEncoder.endEncoding()

        commandBuffer.present(currentDrawable)
        commandBuffer.commit()
    }

    func loadMetal() {
        commandQueue = device.makeCommandQueue()

        let imagePlaneVertexDataCount = kImagePlaneVertexData.count * MemoryLayout<Float>.size
        imagePlaneVertexBuffer = device.makeBuffer(bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: [])
        // カメラキャプチャ画像のキャッシュ
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        capturedImageTextureCache = textureCache
        // カメラキャプチャ画像＋人体画像の合成パイプライン
        let defaultLibrary = device.makeDefaultLibrary()!

        let compositePipelineStateDescriptor = MTLRenderPipelineDescriptor()
        compositePipelineStateDescriptor.sampleCount = 1
        compositePipelineStateDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "compositeImageVertexTransform")!
        compositePipelineStateDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "compositeImageFragmentShader")!
        compositePipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        try! compositePipelineState = device.makeRenderPipelineState(descriptor: compositePipelineStateDescriptor)

        // 人体縁取り用コンピュートシェーダー
        let edgeShader = defaultLibrary.makeFunction(name: "matteConvert")!
        computeState = try! self.device.makeComputePipelineState(function: edgeShader)
    }

    // キャプチャイメージからMTLTextureを生成
    func createTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)

        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, capturedImageTextureCache, pixelBuffer, nil, pixelFormat,
                                                               width, height, planeIndex, &texture)
        if status != kCVReturnSuccess {
            texture = nil
        }
        return texture
    }

    func compositeImagesWithEncoder(renderEncoder: MTLRenderCommandEncoder) {
        guard let textureY = capturedImageTextureY, let textureCbCr = capturedImageTextureCbCr else { return }
        renderEncoder.setCullMode(.none)
        renderEncoder.setRenderPipelineState(compositePipelineState)

        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 0)

        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureY), index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureCbCr), index: 1)
        renderEncoder.setFragmentTexture(whiteBlurTexture, index: 2)
        renderEncoder.setFragmentTexture(yellowBlurTexture, index: 3)
        renderEncoder.setFragmentTexture(alphaTexture, index: 4)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}
