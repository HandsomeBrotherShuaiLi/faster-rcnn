from keras import backend as K
class Config(object):
    def __init__(self):
        self.verbose=True
        #base network
        self.network='resnet50'
        #setting for data augmentation
        self.use_horizontal_flips=False
        self.use_vertical_flips=False
        self.rot_90=False
        #anchor box ratios
        self.anchor_box_ratios=[[1,1],[1,2],[2,1]]
        #anchor box scales
        self.anchor_box_scales=[128,256,512]
        #size to resize the smallest of the image
        # Mark
        self.im_size=600
        #image channel-wise mean to substract
        self.img_channel_mean=[103.939, 116.779, 123.68]
        self.img_scaling_factor=1.0

        #number of ROIs
        self.num_rois=300
        #it depends on the network(resnet50)
        self.rpn_stride=16
        self.balanced_classes=False

        #scaling the stdev
        self.std_scaling=4.0
        self.classfier_regr_std=[8.0,8.0,4.0,4.0]

        self.rpn_min_overlap=0.3
        self.rpn_max_overlap=0.7

        self.classfier_min_overlap=0.1
        self.classfier_max_overlap=0.5
        self.class_mapping=None
        #weight files
        self.base_net_weight='resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        self.model_path="model_faster_rcnn.hdf5"



