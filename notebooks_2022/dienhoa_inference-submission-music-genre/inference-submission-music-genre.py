#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision.all import *
import torchaudio
from sklearn.model_selection import StratifiedKFold
import librosa
import kornia
from scipy import stats


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# In[ ]:


df_train = pd.read_csv('../input/kaggle-pog-series-s01e02/train.csv')
df_test = pd.read_csv('../input/kaggle-pog-series-s01e02/test.csv')
submission = pd.read_csv('../input/kaggle-pog-series-s01e02/sample_submission.csv')


# In[ ]:


train_path = Path('../input/music-genre-spectrogram-pogchamps/spectograms/train')
test_path = Path('../input/music-genre-spectrogram-pogchamps/spectograms/test')


# In[ ]:


def get_y(filename):
    resample_name = filename.stem + '.ogg'
    return df_train[df_train['filename']==resample_name]['genre'].values[0]


# In[ ]:


# Excluded unusual music thanks to this thread: https://www.kaggle.com/c/kaggle-pog-series-s01e02/discussion/312842
def get_items(path): 
    excluded_files = ["010449.png" , 
                      "005589.png" , 
                      "004921.png", 
                      "019511.png" , 
                      "013375.png" , 
                      "024247.png", 
                      "024156.png"]
    items = get_image_files(path)
    items = [item for item in items if item.name not in excluded_files]
    
    ## For fast iteration
#     items = [item for item in items if get_y(item) in ['Punk', 'Rock']]
    random.shuffle(items)
#     items.shuffle()
    return L(items)


# In[ ]:


test_items = get_items(test_path)


# In[ ]:


class ReflectionCrop(RandomCrop):
    def encodes(self, x:(Image.Image,TensorBBox,TensorPoint)):
        return x.crop_pad(self.size, self.tl, orig_sz=self.orig_sz, pad_mode=PadMode.Reflection)


# In[ ]:


class CustomDataBlock(DataBlock):
    def datasets(self:DataBlock, source, verbose=False, splits=None):
        self.source = source                     ; pv(f"Collecting items from {source}", verbose)
        items = (self.get_items or noop)(source) ; pv(f"Found {len(items)} items", verbose)
        pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
    def dataloaders(self, source, path='.', verbose=False, splits=None, **kwargs):
        dsets = self.datasets(source, verbose=verbose, splits=splits)
        kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
        return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)


# In[ ]:


def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = kornia.contrib.MaxBlurPool2d(3, True)
            model._modules[name] = layer_new

    return model


# In[ ]:


learns = [load_learner(learn_pkl, cpu=False) for learn_pkl in Path('../input/inference-music-genre').ls()]


# In[ ]:


len(learns)


# In[ ]:


_before_epoch = [event.before_fit, event.before_epoch]
_after_epoch  = [event.after_epoch, event.after_fit]


# In[ ]:


@patch
def ttacustom(self:Learner, ds_idx=1, dl=None, n=4, item_tfms=None, batch_tfms=None, beta=0.25, use_max=False):
    "Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation"
    if dl is None: dl = self.dls[ds_idx].new(shuffled=False, drop_last=False)
    if item_tfms is not None or batch_tfms is not None: dl = dl.new(after_item=item_tfms, after_batch=batch_tfms)
    try:
        self(_before_epoch)
        with dl.dataset.set_split_idx(0), self.no_mbar():
            if hasattr(self,'progress'): self.progress.mbar = master_bar(list(range(n)))
            aug_preds = []
            for i in self.progress.mbar if hasattr(self,'progress') else range(n):
                self.epoch = i #To keep track of progress on mbar since the progress callback will use self.epoch
                preds = self.get_preds(dl=dl, inner=True)[0][None]
                preds_idx = preds.squeeze().argmax(1)
                aug_preds.append(preds_idx)
#         aug_preds = torch.cat(aug_preds)
#         aug_preds = aug_preds.max(0)[0] if use_max else aug_preds.mean(0)
#         self.epoch = n
#         with dl.dataset.set_split_idx(1): preds,targs = self.get_preds(dl=dl, inner=True)
    finally: self(event.after_fit)

#     if use_max: return torch.stack([preds, aug_preds], 0).max(0)[0],targs
#     preds = (aug_preds,preds) if beta is None else torch.lerp(aug_preds, preds, beta)
    return aug_preds


# In[ ]:


aug_preds = []
for learn in learns:
    test_dl = learn.dls.test_dl(test_items)
    learn.dls.bs = 512
    aug_preds_1fold = learn.ttacustom(dl=test_dl, n=50, beta=None)
    aug_preds.extend(aug_preds_1fold)


# In[ ]:


final_votes = stats.mode(torch.vstack(aug_preds))[0][0]


# In[ ]:


torch.vstack(aug_preds).shape


# In[ ]:


final_votes.shape


# In[ ]:


def genreid_from_genre(genre):
    return int(genre2id[genre2id['genre'] == genre]['genre_id'].values[0])


# In[ ]:


# preds_idx = final_preds.argmax(axis=1)
genre2id = pd.read_csv('../input/kaggle-pog-series-s01e02/genres.csv')
songid_preds = {int(file_path.stem):genreid_from_genre(learns[0].dls.vocab[_id]) for file_path, _id in zip(test_items,final_votes)}
submission['genre_id'] = submission['song_id'].map(songid_preds)
submission['genre_id'].fillna(0, inplace=True)
submission.loc[submission['song_id']==22612, 'genre_id'] = 1
submission.loc[submission['song_id']==24013, 'genre_id'] = 0

submission.genre_id = submission.genre_id.astype(int)
submission.to_csv(f"submission_final_{int(time.time())}.csv", index=False)


# In[ ]:




