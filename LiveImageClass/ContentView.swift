//
//  ContentView.swift
//  LiveImageClass
//
//  Created by Heical Chandra on 30/09/24.
//

import SwiftUI
import AVKit
import CoreML
import Vision

struct ContentView: View {
    @StateObject private var viewModel = ImageClassificationViewModel()

    var body: some View {
        VStack {
            Text("Image Classification")
                .font(.title)
                .padding()

            if let image = viewModel.capturedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
            } else {
                Rectangle()
                    .fill(Color.gray)
                    .frame(height: 300)
            }

            Text(viewModel.classificationLabel)
                .font(.headline)
                .padding()

            // Hanya tombol rotate camera
            Button(action: {
                viewModel.toggleCamera()
            }) {
                Text("Rotate Camera")
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
        }
        .onAppear {
            viewModel.setupCamera()
        }
    }
}



class ImageClassificationViewModel: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var classificationLabel: String = "Waiting for image..."

    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let mlModel = try! FishShrimpClassificationV1().model
    private var request: VNCoreMLRequest!
    private var currentCamera: AVCaptureDevice.Position = .back  // Default to back camera

    override init() {
        super.init()
        setupMLModel()
    }

    func setupMLModel() {
        guard let model = try? VNCoreMLModel(for: mlModel) else {
            fatalError("Failed to load CoreML model.")
        }
        request = VNCoreMLRequest(model: model, completionHandler: handleClassification)
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let camera = camera(with: currentCamera),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return
        }

        if captureSession.inputs.isEmpty {
            captureSession.addInput(input)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)

        captureSession.startRunning()
    }

    func startCamera() {
        captureSession.startRunning()
    }

    func toggleCamera() {
        captureSession.stopRunning()
        captureSession.inputs.forEach { input in
            captureSession.removeInput(input)
        }

        currentCamera = (currentCamera == .back) ? .front : .back

        setupCamera()
    }

    private func camera(with position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let devices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: position
        ).devices

        return devices.first
    }

    private func handleClassification(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNClassificationObservation],
              let bestResult = results.first else {
            classificationLabel = "Unable to classify image."
            return
        }

        DispatchQueue.main.async { [weak self] in
            self?.classificationLabel = "\(bestResult.identifier) - \(String(format: "%.2f", bestResult.confidence * 100))%"
        }
    }

    private func classifyImage(_ image: UIImage) {
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to convert UIImage to CIImage.")
        }

        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform classification: \(error)")
        }
    }
}

extension ImageClassificationViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait  // Set to portrait mode
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)

        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let uiImage = UIImage(cgImage: cgImage)
            DispatchQueue.main.async { [weak self] in
                self?.capturedImage = uiImage
                self?.classifyImage(uiImage)
            }
        }
    }
}
