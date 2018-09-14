# FirebaseML-TextRecognition

#### [Add](https://firebase.google.com/docs/ios/setup) Firebase to your app.

#### Add pods 

```bash
pod 'Firebase/Core'
pod 'Firebase/MLVision'
# If using an on-device API:
pod 'Firebase/MLVisionTextModel'
```

#### Enum for API state wheter it's on-device API or cloud-based API

```swift
enum APIState {
    case onDevice
    case cloudBased
}
```

#### Write the text detection method

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
