# Model settings
model = dict(
    encoder = 'SegFormer',
    decoder= 'PATHFinder',
    device = 'cuda'
    pretrained = 'weights/pathfinder.pth'
)

# Dataset settings
dataset = dict(
    type='DeepGlobe',  # Dataset type
    root_dir='',  # Root directory of deepglobe dataset
    crop_size=(512, 512),  # Image crop size during evaluating
    batch_size = 8, # batch size
)










