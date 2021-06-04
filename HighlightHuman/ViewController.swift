//
//  ViewController.swift
//  HighlightHuman
//
//  Created by Higashihara Yoki on 2021/06/04.
//

import UIKit
import Metal
import MetalKit
import ARKit


class ViewController: UIViewController, MTKViewDelegate {

    var session = ARSession()
    var renderer: Renderer!

    override func viewDidLoad() {
        super.viewDidLoad()

        if let view = self.view as? MTKView {
            view.device = MTLCreateSystemDefaultDevice()
            view.backgroundColor = UIColor.clear
            view.delegate = self
            renderer = Renderer(session: session, metalDevice: view.device!, mtkView: view)
            renderer.drawRectResized(size: view.bounds.size)
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .personSegmentation
        session.run(configuration)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }

    func draw(in view: MTKView) {
        renderer.update()
    }
}
