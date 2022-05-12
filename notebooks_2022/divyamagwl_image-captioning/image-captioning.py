#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

import os
import string
import pickle
import numpy as np 
from PIL import Image 
from matplotlib import pyplot as plt
from tqdm import tqdm
import re


# # Import Dataset

# In[ ]:


img_transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[ ]:


def extract_images(image_dir):    
    images_list = []
    file_names_list = []
    for f in tqdm(os.listdir(image_dir)):
        image = Image.open(image_dir + '/' + f)
        image = img_transform(image).unsqueeze(0)
        images_list.append(image)
        file_names_list.append(f)
        
    return images_list, file_names_list


# In[ ]:


image_path = "../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flicker8k_Images"


# In[ ]:


images_list, file_names_list = extract_images(image_path)


# # Feature Extraction

# In[ ]:


def generate_feature(images, model):
    features = {}

    for i in tqdm(range(len(images))):
        image = images[i]
        file_name = file_names_list[i]

        with torch.no_grad():
            feature = model(image).detach().numpy()
        features[file_name] = feature
        
    return features


# In[ ]:


alexnet_model = models.alexnet(pretrained = True)


# In[ ]:


# resnet_model = models.resnet34(pretrained = True)


# In[ ]:


image_features = generate_feature(images_list, alexnet_model)
with open('alexnet_features.pickle', 'wb') as handle:
    pickle.dump(image_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


with open('alexnet_features.pickle', 'rb') as handle:
    image_features = pickle.load(handle)


# In[ ]:


# image_features = generate_feature(images_list, resnet_model)
# with open('resnet_features.pickle', 'wb') as handle:
#     pickle.dump(image_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# with open('resnet_features.pickle', 'rb') as handle:
#     image_features = pickle.load(handle)


# # Load captions

# In[ ]:


caption_text_path = "../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flickr8k_text/Flickr8k.token.txt"


# In[ ]:


with open(caption_text_path, "r") as f:
    captions = f.read().split("\n")


# In[ ]:


captions[0]


# In[ ]:


def generate_captions_dict(captions):
    captions_dict = {}

    for line in captions:
        contents = line.split("\t")
        if len(contents) < 2:
            continue

        filename, caption = contents[0][:-2], contents[1]

        if filename in captions_dict.keys():
            captions_dict[filename].append(caption)
        else:
            captions_dict[filename] = [caption]
            
    return captions_dict


# In[ ]:


captions_dict = generate_captions_dict(captions)


# # Preprocessing on Captions

# In[ ]:


def preprocess_sentence(line, lower = True, remove_num = True, remove_punct = True):
    line = line.split()
    if lower:
        line = [word.lower() for word in line]
    if remove_num:
        line = [word for word in line if word.isalpha()]
    if remove_punct:
        line = [re.sub(r'[^\w\s]', '', word) for word in line]
    line = ' '.join(line)
    line = "begin " + line + " end"
    return line


# In[ ]:


def tokenize_sentences(captions_dict):
    all_captions = []
    for captions_list in captions_dict.values():
        all_captions.extend(captions_list)
    
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


# In[ ]:


for _, caption_list in captions_dict.items():
    for i in range(len(caption_list)):
        caption_list[i] = preprocess_sentence(caption_list[i])


# In[ ]:


tokenizer, vocab_size = tokenize_sentences(captions_dict)


# # Train and test data

# In[ ]:


def extract_test_train_data(path, features_dict, captions_dict):
    with open(path, "r") as f:
        file_names = f.read().split("\n")
    
    features_data = {}
    captions_data = {}

    for name in file_names:
        if name in features_dict.keys() and name in captions_dict.keys():
            features_data[name] = features_dict[name]
            captions_data[name] = captions_dict[name]

    return features_data, captions_data


# In[ ]:


train_path = "../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flickr8k_text/Flickr_8k.trainImages.txt"
val_path = "../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flickr8k_text/Flickr_8k.valImages.txt"
test_path = "../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flickr8k_text/Flickr_8k.testImages.txt"


# In[ ]:


train_features, train_captions = extract_test_train_data(train_path, image_features, captions_dict)


# In[ ]:


val_features, val_captions = extract_test_train_data(val_path, image_features, captions_dict)


# In[ ]:


test_features, test_captions = extract_test_train_data(test_path, image_features, captions_dict)


# # Data generator

# In[ ]:


def data_generator(image_features, captions_dict, batch_size, tokenizer):
    img_input, caption_input, caption_target = [], [], []
    
    count = 0
   
    while True:
        for name, caption_list in captions_dict.items():            
            img_ft = image_features[name]
            
            for caption in caption_list:                
                caption_seq = tokenizer.texts_to_sequences([caption])[0]
                
                for i in range(1, len(caption_seq)):
                    in_seq, trg_seq = caption_seq[:i], caption_seq[i]
                    
                    img_input.append(img_ft)
                    caption_input.append(in_seq)
                    caption_target.append(trg_seq)
                    count += 1
                
                    if count == batch_size:
                        caption_input = pad_sequences(caption_input, padding='pre')

                        yield (
                            torch.FloatTensor(img_input).squeeze(1),
                            torch.LongTensor(caption_input),
                            torch.LongTensor(caption_target)
                        )

                        # Reset
                        img_input, caption_input, caption_target = [], [], []                        
                        count = 0


# In[ ]:


generator = data_generator(train_features, train_captions, 32, tokenizer)

img_input, caption_input, caption_target = next(generator)

print("Image features:", img_input.shape)
print("Caption input:", caption_input.shape)
print("Caption target:", caption_target.shape)


# # Glove Dataset

# In[ ]:


with open("../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt", "r") as f:
    glove = f.read().split("\n")


# In[ ]:


glove_dict = {}

for line in glove:
    try:
        elements = line.split()
        word, vector = elements[0], np.array([float(i) for i in elements[1:]])
        glove_dict[word] = vector
    except:
        continue


# In[ ]:


glove_weights = np.random.uniform(0, 1, (vocab_size, 200))
found = 0

for word in tokenizer.word_index.keys():
    if word in glove_dict.keys():
        glove_weights[tokenizer.word_index[word]] = glove_dict[word]
        found += 1
    else:
        continue
        
print("Number of words found in GloVe: {} / {}".format(found, vocab_size))


# # CNN + LSTM Model

# In[ ]:


class Network(torch.nn.Module):

    def __init__(self, glove_weights):
        super(Network, self).__init__()

        self.fc_img = torch.nn.Linear(1000, 512)
        self.embedding = torch.nn.Embedding(vocab_size, 200)
        self.lstm = torch.nn.LSTM(200, 512, batch_first=True)
        self.fc_wrapper = torch.nn.Linear(1024, 1024)
        self.fc_output = torch.nn.Linear(1024, vocab_size)

        self.embedding.weight = torch.nn.Parameter(glove_weights)


    def forward(self, input_img, input_caption):
        x1 = F.tanh(self.fc_img(input_img))

        x2, _ = self.lstm(self.embedding(input_caption))
        x2 = x2[:, -1, :].squeeze(1)

        x3 = torch.cat((x1, x2), dim=-1)
        x3 = F.tanh(self.fc_wrapper(x3))

        x4 = self.fc_output(x3)

        out = F.log_softmax(x4, dim=-1)
        return out


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


epochs = 20
steps_per_epoch = len(train_captions)

model = Network(glove_weights=torch.FloatTensor(glove_weights))
optimizer = optim.Adam(model.parameters(), lr=0.0005)

if torch.cuda.is_available():
    model = model.cuda()

for epoch in tqdm(range(epochs)):
    print("Epoch {}".format(epoch+1))
    
    d_gen = data_generator(train_features, train_captions, 32, tokenizer)
    total_loss = 0
    
    for batch in range(steps_per_epoch):
        img_in, caption_in, caption_trg = next(d_gen)
        img_in = img_in.to(device)
        caption_in = caption_in.to(device)
        caption_trg = caption_trg.to(device)
        
        optimizer.zero_grad()
    
        preds = model.forward(img_in, caption_in)
        loss = F.nll_loss(preds, caption_trg)    
        loss.backward()

        optimizer.step()

        total_loss += loss
        
        if batch % 2500 == 0:
            print("Epoch {} - Batch {} - Loss {:.4f}".format(epoch+1, batch, loss))
            
    epoch_loss = total_loss/steps_per_epoch
    
    print("Epoch {} - Average loss {:.4f}".format(epoch+1, epoch_loss))
    
    torch.save(model.state_dict(), "model_{}".format(epoch+1))


# In[ ]:


def translate(features):    
    features = torch.FloatTensor(features)
    
    result = "begin "
    
    for _ in range(max_length):
        in_seq = tokenizer.texts_to_sequences([result])
        in_seq = torch.LongTensor(pad_sequences(in_seq, maxlen=max_length, padding='pre'))
        
        preds = model.forward(features, in_seq)
        pred_idx = preds.argmax(dim=-1).detach().numpy()[0]
        word = tokenizer.index_word.get(pred_idx)
        
        if word is None or word == 'end' or word == " end":
            break
            
        result += word + " "
        
    return " ".join(result.split()[1:])


# In[ ]:


def evaluate_model(feature_dict, caption_dict):
    references = []
    hypotheses = []
    
    for name in tqdm(feature_dict.keys()):
        prediction = translate(feature_dict[name])
        hypotheses.append(prediction.split())

        refs = [caption.split() for caption in caption_dict[name]]
        references.append(refs)
            
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("BLEU-1: {:.4f}".format(bleu_1))
    print("BLEU-2: {:.4f}".format(bleu_2))
    print("BLEU-3: {:.4f}".format(bleu_3))
    print("BLEU-4: {:.4f}".format(bleu_4))


# In[ ]:


max_length = 20

model = Network(glove_weights=torch.FloatTensor(glove_weights))

model.load_state_dict(torch.load("model_20", map_location=torch.device('cpu')))


# In[ ]:


evaluate_model(train_features, train_captions)

evaluate_model(test_features, test_captions)


# In[ ]:


name = list(test_features.keys())[123]
image = Image.open("../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flicker8k_Images/" + name)

plt.imshow(image)
plt.axis('off')
plt.show()

print("[CAPTION]: {}".format(translate(test_features[name])))


# In[ ]:


name = list(test_features.keys())[34]
image = Image.open("../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flicker8k_Images/" + name)

plt.imshow(image)
plt.axis('off')
plt.show()

print("[CAPTION]: {}".format(translate(test_features[name])))


# In[ ]:


name = list(train_features.keys())[32]
image = Image.open("../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flicker8k_Images/" + name)

plt.imshow(image)
plt.axis('off')
plt.show()

print("[CAPTION]: {}".format(translate(train_features[name])))


# In[ ]:


name = list(train_features.keys())[143]
image = Image.open("../input/d/datasets/divyamagwl/flickr8k/Flickr8K/Flicker8k_Images/" + name)

plt.imshow(image)
plt.axis('off')
plt.show()

print("[CAPTION]: {}".format(translate(train_features[name])))


# In[ ]:





# In[ ]:




