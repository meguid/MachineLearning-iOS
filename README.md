# Integrating Machine learning Models in iOS Applications
A trained model is the result of applying a machine learning algorithm to a set of training data. The model makes predictions based on new input data. For example, a model that's been trained on a region's historical house prices may be able to predict a house's price when given the number of bedrooms and bathrooms.

## Core ML
The easiest way and the most straight-forward one, Core ML is an apple framework and can help you to integrate ML models to your app in just a few lines of code. First I recommend this kick-start [course](https://eg.udacity.com/course/core-ml--ud1038) in Core ML.

### Option 1: Ready-To-Use Models [Demo: ImageDetection-Example]
Apple provides several popular, open source [models](https://developer.apple.com/machine-learning/build-run-models) that are already in the Core ML model format. You can download these models - which vary from a 5MBs size to a 500MBs size - and start using them in your app.

Create new Xcode project.

Download [MobileNet Model](https://developer.apple.com/machine-learning/build-run-models) and drag-drop to your project.

Add "Privacy - Photo Library Usage Description" to your Info.plist file.

Add UIImagePickerController for picking images

```swift
let picker = UIImagePickerController()
picker.delegate = self
picker.allowsEditing = false
picker.sourceType = .photoLibrary
//        picker.sourceType = .camera
//        picker.cameraCaptureMode = .photo
self.present(picker, animated: true, completion: nil)

func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
    let chosenImage = info[UIImagePickerControllerOriginalImage] as! UIImage
    predict(image: chosenImage)
}

func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    dismiss(animated: true, completion: nil)
}
```

Predict

```swift
func predict(image: UIImage) {
    if let pixelBuffer = image.pixelBuffer(width: 224, height: 224),
        let prediction = try? model.prediction(image: pixelBuffer) {
        log(prediction.classLabelProbs)
    }
}
```

Log Top Results

```swift
func log(_ results: [String: Double]) {

    let resultsCount = min(5, results.count)
    let selectedResults = Array(results.map { x in (x.key, x.value) }
        .sorted(by: { a, b -> Bool in a.1 > b.1 })
        .prefix(through: resultsCount - 1))

    var s: [String] = []
    for (i, result) in selectedResults.enumerated() {
        s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, result.0, result.1 * 100))
    }
    print(s.joined(separator: "\n\n"))
}
```

### Option 2: Convert Open Source Models [Demo: CoreML-Converter]
Various research groups and universities publish their models and training data, which may not be in the Core ML model format. To use these models in your app, you need to convert them, as described in [Converting Trained Models to Core ML](https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml). Or you can use one of this [already converted models](https://github.com/likedan/Awesome-CoreML-Models).

[Download](https://github.com/shicai/MobileNet-Caffe) "mobilenet.caffemodel", "mobilenet_deploy.prototxt" and "synset.txt"

Install Tools

```bash
virtualenv -p /usr/bin/python2.7 env
source env/bin/activate
pip install tensorflow
pip install keras==1.2.2
pip install coremltools
```

Run Script

```bash
python control.py
```
Deactivate from virtual environment

```bash
deactivate
```

### Option 3: Create Your Models [Demo: CreateML-Example]
Using [Create ML](https://developer.apple.com/documentation/createml) with familiar tools like Swift and macOS playgrounds and your own data, you can train custom models on your Mac to perform tasks like recognizing images, extracting meaning from text, or finding relationships between numerical values. Models trained using Create ML are in the Core ML model format and are ready to use in your app.

Download Images about your to be classified objects, for example "Giraffe" and "Elephant".

Create two folders named "Training Data" and "Testing Data" and put your images in them with subfolders named after the object name.

The best practice is to use more images for traning than testing, Apple suggests 80%-20%.

Create a new playground project with the following code.

```swift
import CreateMLUI 

let builder = MLImageClassifierBuilder()
builder.showInLiveView()
```

Run the playground, then add your training and testing results.

Save your CoreMl model and add it to your project.

## Firebase ML Kit
ML Kit lets you bring powerful machine learning features to your app whether it's for Android or iOS, and whether you're an experienced machine learning developer or you're just getting started.

### Option 1: Ready-To-Use APIs [Demo: FirebaseML-TextRecognition]

[Add](https://firebase.google.com/docs/ios/setup) Firebase to your app.

Add pods 

```bash
pod 'Firebase/Core'
pod 'Firebase/MLVision'
# If using an on-device API:
pod 'Firebase/MLVisionTextModel'
```

Enum for API state wheter it's on-device API or cloud-based API

```swift
enum APIState {
    case onDevice
    case cloudBased
}
```

Write the text detection method

```swift
let state: APIState = .onDevice
    
func detect(image: UIImage) {

    let vision = Vision.vision()

    let onDeviceTextRecognizer = vision.onDeviceTextRecognizer()
    let cloudBasedTextRecognizer = vision.cloudTextRecognizer()

    let textRecognizer = state == .onDevice ? onDeviceTextRecognizer : cloudBasedTextRecognizer

    let visionImage = VisionImage(image: image)

    textRecognizer.process(visionImage) { result, error in
        guard error == nil, let result = result else {
            print("Recognied Error: \(error)")
            return
        }
        let resultText = result.text
        print("Recognied text: \(resultText)")
    }
}
```
### Option 2: Deploy Custom Models [Demo: FirebaseML-TensorFlowLite]

[Add](https://firebase.google.com/docs/ios/setup) Firebase to your app.

Add pods.

```bash
pod 'Firebase/Core'
pod 'Firebase/MLModelInterpreter'
```

Add your .tflite custom model on [Firebase Console](https://console.firebase.google.com/u/0/) or to your bundle resources.

Enum for API state wheter it's on-device API or cloud-based API

```swift
enum APIState {
    case onDevice
    case cloudBased
}
```

Load the .tflite model

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

Set Input and Output Options/Format

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

Run the predict method

```swift
let interpreter = ModelInterpreter(options: ModelOptions(cloudModelName: "my_cloud_model", localModelName: "my_local_model"))

interpreter.run(inputs: input, options: ioOptions) { outputs, error in
    guard error == nil, let outputs = outputs else { return }
        let probabilities = try? outputs.output(index: 0)
        print(probabilities)
}
```
