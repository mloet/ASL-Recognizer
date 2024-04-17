import os
import torch
from slgcn_model import GCN_muti_att
from pytorch_i3d import InceptionI3d
from stream import from_stream
from record import from_recording
from read_mp4 import from_video

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

  tgcn = GCN_muti_att(input_feature=100, hidden_feature=64, num_class=100, p_dropout=0.3, num_stage=20)
  tgcn.load_state_dict(torch.load(os.path.join('model', '100gcn.pth'), map_location=torch.device('cpu')))
  tgcn.eval()
  with open('labels.txt', 'r') as file:
    labels = list(map(str.strip, file.readlines()))
  
  if mode == 'stream':
    print('Press \'q\' to quit')
    sentence = from_stream(tgcn, labels)
  elif mode == 'video':
    while True:
      vid_path = input('\nPlease enter path to desired mp4 file, or \'quit\': ')
      if os.path.exists(vid_path):
        sentence = from_video(vid_path, i3d, labels)
        break
      elif vid_path == 'quit':
        break
    
  elif mode == 'record':
    print('Press space to start/stop recording, and \'q\' to quit')
    sentence = from_recording(i3d, labels)

  print(' '.join(sentence))