# Machine-Learning-iOS-Tutorial
This is the sourcecode used for "Integrating Machine learning Models in iOS Applications"

## Core ML

### 1 - ImageDetection-Example

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

### 2 - CoreML-Converter

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

### 3 - CreateML-Example

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
