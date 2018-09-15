//
//  ViewController.swift
//  FirebaseML-TextRecognition
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
}
