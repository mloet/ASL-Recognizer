import cv2
import numpy as np
import torch
import mediapipe as mp
import transformers 
from stream_utils import extract_keypoints_3d

def from_stream(cmodel, labels):
  fill_mask = transformers.pipeline("fill-mask", model="distilroberta-base")
  tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
  nlp = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
  display_sentence = ""
  sequence = []
  sentence = []
  threshold = 0.9
  prev = np.zeros((55, 3))
  mp_holistic = mp.solutions.holistic 
  cap = cv2.VideoCapture(0)
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      success, frame = cap.read()
      
      image = frame
      w, h, c = image.shape

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False                  
      results = holistic.process(image)                 
      image.flags.writeable = True                   
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

      keypoints = extract_keypoints_3d(results, prev)
      prev = keypoints

      sequence.append(torch.FloatTensor((keypoints[:, (0, 1)]-0.5)*2))
      sequence = sequence[-50:]
      
      if len(sequence) == 50:
        input = torch.cat(sequence,  dim = 1).unsqueeze(0)
        per_frame_logits = cmodel(input)
        p, k  = torch.nn.functional.softmax(per_frame_logits, dim = 1).topk(10, dim = 1)
        if p[0,0].item()>threshold: 
          word = labels[k[0,0].item()]

          if len(sentence) > 0: 
            input_ids = tokenizer.encode(display_sentence, return_tensors="pt")
            output = nlp.generate(input_ids, max_length=len(input_ids[0])+1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
            predicted_word = tokenizer.decode(output[0][-1], skip_special_tokens=True)[1:]
            for i, n in enumerate(k[0]):
              if labels[n.item()]==predicted_word:
                word = predicted_word
            if word != sentence[-1]:
              sentence.append(word)
              fill = fill_mask(display_sentence + f' <mask> {word}')[0]
              display_sentence = fill['sequence'] if fill['score'] > 0.05 else f"{display_sentence} {word}"
          else:
            sentence.append(word)
            display_sentence = word
            
      cv2.rectangle(image, (0,0), (640, 40), (0, 0, 0), -1)
      cv2.putText(image, display_sentence, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.imshow('OpenCV Feed', image)

      key = cv2.waitKey(10)
      if key & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()
  return sentence