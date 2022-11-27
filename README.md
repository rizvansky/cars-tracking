# Vehicle recognition in video surveillance footage

Vehicle Make and Model Recognition (VMMR) is gaining popularity owing to the current security state in Intelligent transport system (ITS), since it plays a vital role in surveillance and forensics. The primary problem that has to be solved is the need that the proposed VMMR algorithm be able to discriminate between various manufacturers and models while categorizing various variations of a model as one. The Histogram of Oriented Gradient (HOG), features from accelerated segment test (FAST) and continuous wavelet transform (CWT) are used to extract the features from the images in this project together with YOLOv5 deep neural network to address the VMMR issue.

---

The adopted pipeline in this project is consist of three main parts: vehicle detection, feature extraction and vehicle classification. The vehicle detection is performed using YOLOv5 deep neural network. The features are extracted from the detected vehicles using HOG, CWT and FAST. The features are then used to train the SVM classifier to classify the vehicles into different classes. The proposed pipeline is evaluated on the [VRiV (Vehicle Recognition in Videos) Datas]](https://www.kaggle.com/c/vehicle-make-model-recognition/overview) and the results are compared with the baseline model. Next subsections describe the project in more detail. 

### Vehicle detection
The YOLOv5 deep neural network for detection was utilized. The CSP-Darknet53 (Cross-stage Partial network) was used as a backbone in order to obtain information and important features from input images. For the Neck layer the PANet (Path Aggregation Network) was utilized providing an ability to recognize the same object in various sizes and scales. The YOLOv3 head was adopted for the final detection step.

### Feature extraction
Histogram of oriented gradients (HOG), Features from accelerated segment test (FAST), Continuous wavelet transform (CWT) were tested for feature extraction. Through experiments the feature descriptor HOG is most suitable method for vehicle classification.

### Classification
The classification of the vehicle make and model was performed with the support vector machine (SVM) trained on the extracted features and achieve the accuracy of 0.99 with HOG features.

### Colaborators
- Daniil Igudesman (d.igudesman@innopolis.university)
- Rizvan Iskaliev (r.iskaliev@innopolis.university)
- Lada Morozova (l.morozova@innopolis.university)