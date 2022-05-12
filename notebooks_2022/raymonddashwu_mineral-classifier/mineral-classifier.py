#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai -Uqq')


# In[ ]:


from fastai.vision.all import *

def search_images(term, max_images=100):
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        data = urljson(requestUrl,data=params)
        urls.update(L(data['results']).itemgot('image'))
        requestUrl = url + data['next']
        time.sleep(0.2)
    return L(urls)[:max_images]


# In[ ]:


# Moh's hardness + glass for now. TODO scrape Reddit r/whatsthisrock/ with image + top answer
searches = 'glass slag', 'talc -powder', 'gypsum -powder', 'calcite', 'fluorite', 'apatite', 'feldspar', 'quartz', 'topaz -facet -gem -gemstone -carat -cut -jewelry', 'corundum'
path = Path('mineral_classifier')

for o in searches:
    dest = (path/o.strip())
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    resize_images(path/o, max_size=400, dest=path/o.strip())


# In[ ]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink);


# In[ ]:


minerals = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[ ]:


minerals = minerals.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = minerals.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)


# In[ ]:


minerals = minerals.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = minerals.dataloaders(path)


# In[ ]:


# for initial training to determine what needs to be thrown out. High error rate
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


from fastai.vision.widgets import *
cleaner = ImageClassifierCleaner(learn)
cleaner


# In[ ]:


for idx in cleaner.delete(): cleaner.fns[idx].unlink()


# In[ ]:


lr_valley,lr_steep = learn.lr_find(suggest_funcs=(valley, steep))


# In[ ]:


lr_valley


# In[ ]:


learn.fit_one_cycle(2, lr_valley)


# In[ ]:


learn.unfreeze()


# In[ ]:


lr_valley,lr_steep = learn.lr_find(suggest_funcs=(valley, steep))


# In[ ]:


learn.fit_one_cycle(15, lr_max=slice(lr_valley, lr_steep))


# In[ ]:




