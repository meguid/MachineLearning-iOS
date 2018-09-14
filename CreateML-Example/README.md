# CreateML-Example

#### Download Images about your to be classified objects, for example "Giraffe" and "Elephant".

#### Create two folders named "Training Data" and "Testing Data" and put your images in them with subfolders named after the object name.

#### The best practice is to use more images for traning than testing, Apple suggests 80%-20%.

#### Create a new playground project with the following code.

```swift
import CreateMLUI 

let builder = MLImageClassifierBuilder()
builder.showInLiveView()
```

#### Run the playground, then add your training and testing results.

#### Save your CoreMl model and add it to your project.
