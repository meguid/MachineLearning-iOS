# FirebaseML-TensorFlowLite

#### [Add](https://firebase.google.com/docs/ios/setup) Firebase to your app.

#### Add pods.

```bash
pod 'Firebase/Core'
pod 'Firebase/MLModelInterpreter'
```

#### Add your .tflite custom model on [Firebase Console](https://console.firebase.google.com/u/0/) or to your bundle resources.

#### Enum for API state wheter it's on-device API or cloud-based API

```swift
enum APIState {
    case onDevice
    case cloudBased
}
```

#### Load the .tflite model

```swift
let state: APIState = .onDevice
    
func loadModel() {
    if state == .onDevice {
        ModelManager.modelManager().register(loadModelFromLocal())
    } else {
        ModelManager.modelManager().register(loadModelFromCloud())
    }
}

private func loadModelFromCloud() -> CloudModelSource {
    let conditions = ModelDownloadConditions(isWiFiRequired: true, canDownloadInBackground: true)
    return CloudModelSource(
        modelName: "my_cloud_model",
        enableModelUpdates: true,
        initialConditions: conditions,
        updateConditions: conditions
    )
}

private func loadModelFromLocal() -> LocalModelSource {
    guard let modelPath = Bundle.main.path(
        forResource: "my_model",
        ofType: "tflite"
        ) else {
            return
    }
    return LocalModelSource(modelName: "my_local_model", path: modelPath)
}
```

#### Set Input and Output Options/Format

```swift
func run() {
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
}
```

#### Run the predict method

```swift
let interpreter = ModelInterpreter(options: ModelOptions(cloudModelName: "my_cloud_model", localModelName: "my_local_model"))

interpreter.run(inputs: input, options: ioOptions) { outputs, error in
    guard error == nil, let outputs = outputs else { return }
        let probabilities = try? outputs.output(index: 0)
        print(probabilities)
}
```
