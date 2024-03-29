//
//  ViewController.swift
//  ImageDetection-Example
//
//  Created by Ahmed Abdel Meguid on 9/3/18.
//  Copyright © 2018 Ahmed Abdel Meguid. All rights reserved.
//
import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    let picker = UIImagePickerController()
    let model = MobileNet()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        picker.delegate = self
        picker.allowsEditing = false
        picker.sourceType = .photoLibrary
        //        picker.sourceType = .camera
        //        picker.cameraCaptureMode = .photo
        self.present(picker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let chosenImage = info[UIImagePickerControllerOriginalImage] as! UIImage
        predict(image: chosenImage)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func predict(image: UIImage) {
        if let pixelBuffer = image.pixelBuffer(width: 224, height: 224),
            let prediction = try? model.prediction(image: pixelBuffer) {
            log(prediction.classLabelProbs)
        }
    }
    
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
}
