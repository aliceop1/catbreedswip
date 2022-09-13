from fastai.vision.all import *
import gradio as gr
import fastbook
fastbook.setup_book()
from fastbook import *
from ipywidgets import *
import streamlit as st

path = Path()

# Download images of different bear categories
def download_images_ddg(img_category, img_types, num_images):
    if not path.exists():
        path.mkdir()
        for o in img_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_ddg(f'{str(o)} {img_category}', max_images=num_images)
            for u in range(len(results)):
                try:
                    download_url(url=results[u], 
                                 dest=f'{dest}/{str(o)}-{str(u+1)}.jpg', 
                                 timeout=400, 
                                 show_progress=False)
                except:
                    print(f'not found {results[u]}')
                    continue

# Define image category and image types
img_category = 'cat'
img_types = ['bengal','siamese','ragdoll', 'maine coon', 'persian', 'british shorthair', 'sphynx', 'abyssinian', 'scottish fold',
'birman', 'russian blue', 'calico', 'norwegian forest cat', 'manx', 'havana brown', 'highlander', 'cornish rex', 'egyptian mau', 
'ragamuffin', 'oriental shorthair', 'american shorthair', 'moggie']
path = Path(img_category)

# Call function to download images
download_images_ddg(img_category, img_types, num_images=100)

# Check for failed images
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])


cats = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=1),
    get_y=parent_label,
    item_tfms=Resize(128))
    
dls = cats.dataloaders(path)

cats = cats.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())


dls = cats.dataloaders(path)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
learn.export('catbreeds.pkl')

learn_inf = load_learner('catbreeds.pkl')



#learn_inf = load_learner(path/'export.pkl')
#learn_inf.predict('images/bengal.jpg')







