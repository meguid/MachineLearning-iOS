//
//  ViewController.swift
//  FirebaseML-TensorFlowLite
//
//  Created by Ahmed Abdel Meguid on 9/15/18.
//  Copyright © 2018 Ahmed Abdel Meguid. All rights reserved.
//

import UIKit
import Firebase

enum APIState {
    case onDevice
    case cloudBased
}

class ViewController: UIViewController {

    let state: APIState = .onDevice

    func run() {
        
        let interpreter = ModelInterpreter(options: ModelOptions(cloudModelName: "my_cloud_model", localModelName: "my_local_model"))
        
        let ioOptions = ModelInputOutputOptions()
        do {
            try ioOptions.setInputFormat(index: 0, type: .uInt8, dimensions: [1, 640, 480, 3])
            try ioOptions.setOutputFormat(index: 0, type: .float32, dimensions: [1, 1000])
        } catch let error as NSError {
            print("Failed to set input or output format with error: \(error.localizedDescription)")
        }
        
        let input = ModelInputs()
        do {
            var data: Data
            try input.addInput(data)
        } catch let error as NSError {
            print("Failed to add input: \(error.localizedDescription)")
        }
        
        interpreter.run(inputs: input, options: ioOptions) { outputs, error in
            guard error == nil, let outputs = outputs else { return }
                let probabilities = try? outputs.output(index: 0)
                print(probabilities)
        }
    }
    
    func setModel() {
        if state == .onDevice {
            ModelManager.modelManager().register(getModelFromLocal())
        } else {
            ModelManager.modelManager().register(getModelFromCloud())
        }
    }
    
    private func getModelFromCloud() -> CloudModelSource {
        let conditions = ModelDownloadConditions(isWiFiRequired: true, canDownloadInBackground: true)
        return CloudModelSource(
            modelName: "my_cloud_model",
            enableModelUpdates: true,
            initialConditions: conditions,
            updateConditions: conditions
        )
    }
    
    private func getModelFromLocal() -> LocalModelSource {
        guard let modelPath = Bundle.main.path(
            forResource: "my_model",
            ofType: "tflite"
            ) else {
                return
        }
        return LocalModelSource(modelName: "my_local_model", path: modelPath)
    }
}

