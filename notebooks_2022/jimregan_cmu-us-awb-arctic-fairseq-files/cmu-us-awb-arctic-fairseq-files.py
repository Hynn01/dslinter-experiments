#!/usr/bin/env python
# coding: utf-8

# In[ ]:


RAWTEXT = "../input/cmu-us-awb-arctic-tts-dataset/cmu_us_awb_arctic/etc/txt.done.data"


# In[ ]:


NORMS = {
    "0.75": "zero point seven five",
    "t.h": "t h",
    "1880": "eighteen eighty",
    "16": "sixteenth",
    "1908": "nineteen oh eight",
    "18": "eighteenth",
    "17": "seventeenth",
    "29th": "twenty ninth",
    "mrs": "misses",
    "etc": "etcetera",
    "etc.": "etcetera",
    "to-day": "today",
    "to-day's": "today's",
    "to-morrow": "tomorrow"
}


# In[ ]:


def _check_apos(word):
    if word.endswith("'s"):
        return word
    elif word.endswith("s'"):
        return word
    elif word.endswith("'d"):
        return word
    elif word.endswith("'ve"):
        return word
    elif word.endswith("'re"):
        return word
    elif word.endswith("'ll"):
        return word
    elif word.endswith("n't"):
        return word
    elif word.endswith("'ve"):
        return word
    elif word in ["i'm", "'em", "o'brien"]:
        return word
    else:
        return word.replace("'", "")

def fix_apos(text):
    words = [_check_apos(w) for w in text.split(" ")]
    return " ".join(words)


# In[ ]:


def normalise(text):
    if text[-1] == ".":
        text = text[:-1]
    text = text.lower()
    words = []
    text = text.replace(",", "")
    for word in text.split(" "):
        if word in NORMS:
            words.append(NORMS[word])
        else:
            words.append(word)
    text = " ".join(words)
    text = text.replace(".", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("--", " ")
    text = text.replace("  ", " ")
    text = text.replace(" - ", " ")
    text = text.replace("to- morrow", "tomorrow")
    text = fix_apos(text)
    text = text.replace("-", " ")
    return text.strip().upper()


# In[ ]:


data = {}
with open(RAWTEXT) as inf:
    for line in inf.readlines():
        first_space = line.find(' ')
        first_quote = line.find('"')
        last_quote = line.rfind('"')
        id = line[first_space+1:first_quote].strip()
        text = line[first_quote+1:last_quote]
        data[id] = normalise(text)


# In[ ]:


with open("text.tsv", "w") as of:
    for id in data.keys():
        of.write(f"{id}\t{data[id]}\n")


# In[ ]:


from pathlib import Path
import soundfile as sf

total = 0
WAVPATH = Path("../input/cmu-us-awb-arctic-tts-dataset/cmu_us_awb_arctic/wav/")
with open("frames.tsv", "w") as of:
    for wav in WAVPATH.glob("*.wav"):
        frames, sr = sf.read(str(wav))
        assert sr == 16000
        total += len(frames)
        of.write(f"{wav.stem}.wav\t{len(frames)}\n")
print("Total:", total / 16000)


# In[ ]:


lines = get_ipython().getoutput("wc -l frames.tsv|awk '{print $1}'")
get_ipython().system('tail -n 114 frames.tsv |head -n 57 > test.tsv')
get_ipython().system('tail -n 114 frames.tsv |tail -n 57 > dev.tsv')
get_ipython().system('head -n $((1138-114)) frames.tsv > train.tsv')


# In[ ]:


def do_fairseq(text):
    words = text.split(" ")
    owords = [" ".join(w) for w in words]
    return " | ".join(owords) + " |"


# In[ ]:


for part in ["test", "train", "dev"]:
    ids = []
    with open(f"{part}.ltr", "w") as of, open(f"{part}.tsv") as inf:
        for line in inf.readlines():
            if "\t" in line:
                parts = line.strip().split("\t")
                id = parts[0].replace(".wav", "")
                of.write(do_fairseq(data[id]) + "\n")

