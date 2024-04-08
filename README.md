# ASL-Recognizer
Personal project to create an application to translate ASL to English using video. Run app.py in either folder to test.

### MNIST 
Initial experimentation with MNIST dataset found on Kaggle. Static gesture detection of A-Z letters.

### WLASL
##### Model
Used the WLASL dataset of videos corresponding to 2000 different ASL words. Used transfer learning to train the Inception3D model (which is trained on ImageNet and fine tuned on the kinetics dataset) on the WLASL dataset. Now changed the streaming mode to run on a temporal GCN using mediapipe keypoints to speed up inference when running in real time.

##### App
Currently app has three options for translation: Realtime detection, recording from webcam (better for machines with low computational power), and translation from mp4 file.

### (Potential) Future Improvements
- Improve sentence prediction.
- Web/phone/AR app?
