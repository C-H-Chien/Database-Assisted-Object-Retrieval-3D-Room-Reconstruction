##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 15                # number of epochs
batch_size = 10             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
#image_size = (448, 448)     # size of training images
image_size = (256, 256)     # size of training images
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'furniture_cabinet'   

# -- directories of .ckpt models for each furniture --
furniture_model_dir = './deepverse/FGVC/Furniture-Models/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name