# Vehicle recognition in video surveillance footage

Following project utilized a pipeline based on the feature
description to solve the vehicle make and model problem. The
pipeline consists of the following steps:

### Vehicle detection
The YOLOv5 deep neural network for detection was utilized. The CSP-Darknet53 (Cross-stage Partial network) was used 
as a backbone in order to obtain information and important features from input images. For the Neck layer the
PANet (Path Aggregation Network) was utilized providing an ability to recognize the same object in various
sizes and scales. The YOLOv3 head was adopted for the final detection step.

### Feature extraction
Histogram of oriented gradients (HOG), Features from accelerated segment test (FAST), Continuous wavelet transform (CWT) were tested for
feature extraction. Through experiments the feature descriptor HOG is most suitable method for vehicle classification.

### Classification
The classification of the vehicle make and model was performed with the support vector machine (SVM).