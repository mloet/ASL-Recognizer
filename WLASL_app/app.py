import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import torch
from torchvision import transforms
import videotransforms
from pytorch_i3d import InceptionI3d


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

weights = os.path.join('model', 'FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt')
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2000)
i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))) 
i3d.eval()
with open('labels.txt', 'r') as file:
    labels = list(map(str.strip, file.readlines()))

sequence = []
sentence = []
predictions = []
threshold = 0.5

print('Starting')
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        # Read feed
        ret, frame = cap.read()

        # Bounding box code
        # image, results = mediapipe_detection(frame, holistic)
        # keypoints = extract_keypoints(results)
        # input_img = generate_bounding_box(keypoints, image)
        # display_img = cv2.cvtColor(np.transpose(np.array(input_img), [1, 2, 0]), cv2.COLOR_BGR2RGB)
        # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 0), 4)

        transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((224, 325)),
          transforms.CenterCrop(224),
          transforms.ToTensor()
        ])
        input_img = transform(frame)
        display_img = np.transpose(np.array(input_img), [1, 2, 0])
        sequence.append(input_img)
        sequence = sequence[-30:]

        if len(sequence) == 30:
          input_sequence = torch.Tensor(np.expand_dims(np.transpose(np.array(sequence), [1, 0, 2, 3]), axis=0))
          with torch.no_grad():
            per_frame_logits = i3d(input_sequence)
          predictions = torch.max(per_frame_logits, dim=2)[0]
          out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
          out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        
        # Show to screen
        cv2.imshow('OpenCV Feed', display_img)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()