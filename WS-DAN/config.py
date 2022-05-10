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
tag = 'furniture_cabinet'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
#save_dir = '../CUB_200_2011/ckpt/'
#save_dir = '../Furniture-Models/bin/ckpt/'
#save_dir = '../Furniture-Models/sofa/ckpt/'
#save_dir = '../Furniture-Models/chair/ckpt/'
#save_dir = '../Furniture-Models/table/ckpt/'
#save_dir = '../Furniture-Models/bed/ckpt/'
#save_dir = '../Furniture-Models/bookcase/ckpt/'
#save_dir = '../Furniture-Models/cabinet/ckpt/'
save_dir = '../Furniture-Models/display/ckpt/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = True
eval_ckpt = save_dir + model_name
#eval_savepath = '../CUB_200_2011/visualize/'
#eval_savepath = '../Furniture-Models/bin/visualize/'
#eval_savepath = '../Furniture-Models/sofa/visualize/'
#eval_savepath = '../Furniture-Models/chair/visualize/'
#eval_savepath = '../Furniture-Models/table/visualize/'
#eval_savepath = '../Furniture-Models/bed/visualize/'
#eval_savepath = '../Furniture-Models/bookcase/visualize/'
#eval_savepath = '../Furniture-Models/cabinet/visualize/'
eval_savepath = '../Furniture-Models/display/visualize/'