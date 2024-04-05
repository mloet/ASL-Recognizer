import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import torch
from torchvision import transforms
from pytorch_i3d import InceptionI3d

def vid_to_tensor(video_path, start=0, num=-1):
  vidcap = cv2.VideoCapture(video_path)
  vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
  if num == -1:
    num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

  frames = []
  for offset in range(num):
    success, img = vidcap.read()
    if not success:
      continue

    w, h, c = img.shape
    sc = 224 / w
    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    img = (img / 255.) * 2 - 1
    img = transforms.CenterCrop(224)(torch.Tensor(img.transpose([2, 0, 1])))
    frames.append(img)
  
  return torch.from_numpy(np.expand_dims(np.asarray(frames, dtype=np.float32).transpose([1, 0, 2, 3]), axis = 0))

def from_stream(model, labels):
  sequence = []
  sentence = []
  threshold = 0.05
  spacer = 0
  cap = cv2.VideoCapture(0)
  
  while cap.isOpened():
    success, frame = cap.read()
    
    img = frame
    w, h, c = img.shape
    sc = 224 / w
    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    img = (img / 255.) * 2 - 1
    img = transforms.CenterCrop(224)(torch.Tensor(img.transpose([2, 0, 1])))
    sequence.append(img)
    sequence = sequence[-60:]
    spacer = (spacer + 1)%5

    if len(sequence) == 60 and spacer == 0:
      input = torch.from_numpy(np.expand_dims(np.asarray(sequence, dtype=np.float32).transpose([1, 0, 2, 3]), axis = 0))
      per_frame_logits = model(input)
      predictions = torch.max(per_frame_logits, dim=2)[0]
      p, k = torch.nn.functional.softmax(predictions, dim = 1).topk(1, dim = 1)
      if(p[0,0].item()>threshold):
        sentence.append(labels[k[0,0].item()])

    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('OpenCV Feed', frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()
  return sentence

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
  threshold = 0.05
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
    if end2 > 10 and word != '':
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

def from_video(vid_path, model, labels):
  sentence = []
  threshold = 0.05

  video = vid_to_tensor(vid_path)
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
    if end2 > 10 and word != '':
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

if __name__ == '__main__':
  mode = ''
  vocab = ''
  vid_path = ''
  sentence = []
  print('Welcome to the ASL translator. Choose your mode: ')
  print('stream: Takes real-time input from your webcam.')
  print('record: Allows you to record a video. Currently better for computers with low computational power.')
  print('video:  Translates from a video on your computer.')
  while True:
    mode = input('Please enter \'stream\', \'record\', or \'video\': ')
    if mode in ['stream', 'record', 'video']:
      break
  print('\nChoose your vocabulary. 100 tends to be more accurate than 2000, but has a smaller lexicon:')
  while True:
    vocab = input('Please enter \'100\' or \'2000\': ')
    if vocab in ['100', '2000']:
      vocab = int(vocab)
      break

  weights = os.path.join('model', f'{vocab}words.pt')
  i3d = InceptionI3d(400, in_channels=3)
  i3d.replace_logits(vocab)
  i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))) 
  i3d.eval()
  with open('labels.txt', 'r') as file:
    labels = list(map(str.strip, file.readlines()))
  
  if mode == 'stream':
    print('Press \'q\' to quit')
    sentence = from_stream(i3d, labels)
  elif mode == 'video':
    while True:
      vid_path = input('\nPlease enter path to desired mp4 file, or \'quit\': ')
      if os.path.exists(vid_path):
        print(vid_path)
        break
      elif vid_path == 'quit':
        break
    sentence = from_video(vid_path, i3d, labels)
  elif mode == 'record':
    print('Press space to start/stop recording, and \'q\' to quit')
    sentence = from_recording(i3d, labels)

  print(' '.join(sentence))