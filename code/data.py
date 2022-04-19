"""
"""

import numpy as np
from sklearn.model_selection import train_test_split
import random
import string
#import shapeworld
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import torch


PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

IMAGE_SIZE = 64 # image is (IMAGE_SIZE, IMAGE_SIZE) pixels


def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }
    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok if type(tok) == str else tok.decode('UTF-8')
    return {'w2i': w2i, 'i2w': i2w}


def train_val_test_split(data,
                         val_size=0.1,
                         test_size=0.1,
                         random_state=None):
    """
    Split data into train, validation, and test splits
    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    """
    idx = np.arange(data['imgs'].shape[0])
    idx_train, idx_valtest = train_test_split(idx,
                                              test_size=val_size + test_size,
                                              random_state=random_state,
                                              shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest,
                                         test_size=test_size /
                                         (val_size + test_size),
                                         random_state=random_state,
                                         shuffle=True)
    splits = []
    for idx_split in (idx_train, idx_val, idx_test):
        splits.append({
            'imgs': data['imgs'][idx_split],
            'labels': data['labels'][idx_split],
            'langs': data['langs'][idx_split],
        })
    return splits

def load_raw_data(data_file):
    data = np.load(data_file)
    # Preprocessing/tokenization
    return {
        'imgs': data['imgs'],
        'labels': data['labels'],
        'langs': np.array([t.lower().split() for t in data['langs']], dtype=object)
    }

class ShapeWorld(Dataset):
    def __init__(self, data, vocab, transform=None, use_unseen_concepts_val_set=False):
        if transform:
            self.imgs = data['imgs']
            # need to transpose for the jitter train set
            # UPDATE: now I don't, since I do the transpose in train.py
            #self.imgs = data['imgs'].transpose(0, 1, 3, 4, 2)
        else:
            
            # use this if using the updated val dataset
            if use_unseen_concepts_val_set:
                self.imgs = data['imgs'].transpose(0, 1, 3, 4, 2)
            else:
                self.imgs = data['imgs'].transpose(0, 1, 4, 2, 3)  # Color channel first


        #random_img = self.imgs[423, 1, :, :, :]
        #Image.fromarray(random_img).save("images/test.png")
        #breakpoint()
        #if augment_data:
        #    pass
        self.labels = data['labels']
        self.lang_raw = data['langs']
        self.use_unseen_concepts_val_set = use_unseen_concepts_val_set

        self.transform = transform # callable to perform SimCLR-style augmentation on images
        #breakpoint()
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)

    def __len__(self):
        return len(self.lang_raw)

    def __getitem__(self, i):
        # Reference game format.
        
        if self.transform:
            imgs = self.imgs[i]        

            # Get mask of zero places and the count of it.
            #mask = imgs==0
            #c = np.count_nonzero(mask)

            # Generate noise numbers for count number of times. 
            # This is where vectorization comes into the play.
            #nums = np.random.randint(0, 255, c)

            # Assign back into X
            #imgs[mask] = nums
            #Image.fromarray(imgs[0]).save("images/before-jitter.png")

            img_set_pairs = [self.transform(Image.fromarray(img)) for img in imgs]
            #img_set = self.transform(self.imgs[i])
            
            img_set_a = torch.stack([item[0] for item in img_set_pairs], dim=0)
            img_set_b = torch.stack([item[1] for item in img_set_pairs], dim=0)
            # convert back to numpy array
            
            img_set = (img_set_a, img_set_b)
            #breakpoint()
        else:
            # ONLY DO THIS FOR THE unseen dataset
            if self.use_unseen_concepts_val_set:
                img_set = self.imgs[i].transpose(0, 3, 1, 2)
            else:
                img_set = self.imgs[i]

        #breakpoint()
        label = self.labels[i]
        lang = self.lang_idx[i]

        #modified_1 = (np.array(img_set[0][0]).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        #modified_2 = (np.array(img_set[1][0]).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        #Image.fromarray(modified_1).save("images/after1-jitter.png")
        #Image.fromarray(modified_2).save("images/after2-jitter.png")
        #breakpoint()
        
        return (img_set, label, lang)

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                if type(tok) != str:
                    tok = tok.decode('UTF-8')
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len

class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self,
        #size: int,
        #augmentation: bool = False,
        #return_original_image: bool = False,
        #dataset_name: str = "imagenet",
    ):
        s = 1
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.15 * s, 0.15 * s)
        transformations = [
            transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0)),
            #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            #transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
        ]

        transformations.extend([transforms.ToTensor()])

        self.transform = transforms.Compose(transformations)

        #self.return_original_image = return_original_image
        #if self.return_original_image:
        #    self.original_image_transform = transforms.Compose(
        #        [transforms.Resize(size=(size, size)), transforms.ToTensor()]
        #    )

    def __call__(self, x):
        x_i = self.transform(x)
        x_j = self.transform(x)
        #if self.return_original_image:
        #    return x_i, x_j, self.original_image_transform(x)
        return x_i, x_j