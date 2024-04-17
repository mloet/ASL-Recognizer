# ASL-Recognizer
This is a personal project to create an application to translate ASL to English using computer vision in real time. Run app.py in either folder to test. Currently app has three options for translation: Realtime detection, recording from webcam, and translation from mp4 file.

### WLASL
The models used in the project are finetuned on the WLASL dataset, a collection of videos corresponding to 2000 different ASL words.

##### Inception3D
Used transfer learning to train the Inception3D model (which is trained on ImageNet and fine tuned on the kinetics dataset) on the WLASL dataset. This model is based on the architecture found here: https://github.com/google-deepmind/kinetics-i3d. While the Inception3D model trained on the kinetics dataset has shown great accuracy in action recognition tasks, running inference on videos takes a significant amount of time for devices with low computational power, which could lead to issues when wanting to run real-time action detection on mobile and wearable devices. Currently this model is in use for the recording and mp4 translation modalities.

##### TGCN
There has been success in the utilization of body keypoints (in concert with or in lieu of rgb images) in action classification using graph convulutional networks, which can greatly decrease input tensor size and as a result inference speed. However, using just body keypoints can pose an issue when concerning accuracy compared to the I3D model. The realtime detection modality currently uses a tgcn model based on the architecture found here: https://arxiv.org/pdf/2103.08833v5.pdf, in concert with MediaPipe's holistic model to identify keypoints. I have experimented with a number of techniques to attempt to increase accuracy and better facilitate translation.

1) Camera Regularization: Without training on extensive datasets or clever data augmentation, using body keypoints can pose an issue when the subject is not facing the camera at a consistant angle, due to the lack of context the model can pick up during training. To help combat this, I have implemented code to transform the keypoints such that the x and y values are consitent with the plane defined by the shoulders and hips.

2) Keypoint Memory: While MediaPipe provides an accurate and lightweight model to track keypoints, occationally the model can fail to recognize the body for a frame or two, which can cause problems in the input tensor. To rectify this, I have implemented code to use the previous frame's keypoints to estimate the current location of undetected keypoints by using the difference of nearby points (e.g., an undetected hand is estimated by using the previous frame's handkeypoints added to the vector of the difference between the current frame and previous frame's corresponding wrist keypoint, preserving detected momentum). I have also implemented code to project keypoints that move outside the camera onto the edge of the visible range.

3) NLP: There are two important ways that we can use natural language processing to assist translation. Firstly, the model architecture and training data availiable isn't perfect for the task of ASL recognition, and as such accuracy can suffer. I am currently using GPT2 to predict the next word in the sentence, to help the application decide between potential words detected. ASL also differs from English in the sense that many of the connecting words used in English do not have corresponding signs in ASL: instead, context clues from the sentence are used to convey meaning. To help facilitate translation, I am currently using a mask filling model from HuggingFace to fill in words between detected signs to attempt to better convey meaning.

### MNIST 
Initial experimentation with MNIST dataset found on Kaggle. Static gesture detection of A-Z letters.

### (Potential) Future Improvements
- Improve GCN model accuracy
- Improve sentence prediction
- Web/phone/AR app?
