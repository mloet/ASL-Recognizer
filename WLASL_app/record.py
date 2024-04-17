import cv2
import numpy as np
import torch
import mediapipe as mp
from torchvision import transforms

def from_recording(model, labels):
  sequence = []
  record = False
  cap = cv2.VideoCapture(0)
  
  while cap.isOpened():
    success, frame = cap.read()
    if record:
      img = frame
      w, h, c = img.shape
      sc = 224 / w
      img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
      img = (img / 255.) * 2 - 1
      img = transforms.CenterCrop(224)(torch.Tensor(img.transpose([2, 0, 1])))
      sequence.append(img)

    cv2.imshow('OpenCV Feed', frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord(' '):
      if not record:
        print("Recording...")
        record = True
      else:
        print("End recording")
        break
    if key & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

  video = torch.from_numpy(np.expand_dims(np.asarray(sequence, dtype=np.float32).transpose([1, 0, 2, 3]), axis = 0))

  sentence = []
  threshold = 0.50
  frames = video.shape[2]
  i = 0
  while i < frames:
    end1 = 30 if i + 30 < frames else frames - i
    end2 = 60 if i + 60 < frames else frames - i

    word = ''

    if end1 > 10:
      chunk1 = video[:, :, i:i+end1, :, :]
      per_frame_logits1 = model(chunk1)
      predictions1 = torch.max(per_frame_logits1, dim=2)[0]
      p1, k1 = torch.nn.functional.softmax(predictions1, dim = 1).topk(1, dim = 1)
      if(p1[0,0].item()>threshold):
        i = i+end1
        word = labels[k1[0,0].item()]
    if end2 > 10 and word == '':
      chunk2 = video[:, :, i:i+end2, :, :]
      per_frame_logits2 = model(chunk2)
      predictions2 = torch.max(per_frame_logits2, dim=2)[0]
      p2, k2 = torch.nn.functional.softmax(predictions2, dim = 1).topk(1, dim = 1)
      if(p2[0,0].item()>threshold):
        i = i+end2
        word = labels[k2[0,0].item()]
    if word == '':
      i+=5
    else:
      sentence.append(word)
    
    if(end1 != 30):
      break

  return sentence