import torch.nn as nn

opt = {
    'version': 1, 

    # Dataset
    'dataset': {
        'name': 'cxr-14',
        'meta': {'image_folder':'images_256'},
        'produce': ['pid', 'img', 'label'], # orig|label_bc|label_mc|label_onehot
        'add_healty_as_label':False,

        'augs':{
            'train':[
                {'name':'RandomRotation', 'degrees':10},
                {'name':'RandomResizedCropAndInterpolation', 'size':256, 'scale':(0.8, 1.2), 'ratio':(1.0, 1.0), 'interpolation':'bicubic'},
                #{'name':'RandomResizedCrop', 'size':512, 'scale':(0.8, 1.2), 'ratio':(1.0, 1.0)},
                #{'name':'CenterCrop', 'size': 512},
                #{'name':'RandomHorizontalFlip', 'p':0.5},
                #{'name':'RandomVerticalFlip', 'p':0.5},
                #{'name':'ColorJitter', 'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'hue':0.5},
                {'name':'ToTensor'},
                {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
            ],
            'test':[
                {'name':'Resize', 'size': 224},
                {'name':'ToTensor'},
                {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
            ]
        },

        'workers':8,
        'batch_size': 128
    },

    # Model definition
    # protonet: 'cls_layer': fc|xproto|pproto|var_xproto|var_pproto
    'model': {  
        #'arch':'hydravit',
        #'arch':'protonet',
        'arch': 'r2r_proto',

        #'backbone': {'arch':'vgg16'},
        #'backbone': {'arch':'resnet50', 'stride_layer2':2, 'stride_layer3':2, 'stride_layer4':2},
        #'backbone': {'arch':'densenet201', 'transition_strides':[2,2,2,2]}, 
        #'backbone': {'arch': 'swin_t'},
        #'backbone': {'arch': 'maxvit_t'},
        
        'num_classes':14,

        'context_encoder': {
            'arch': 'r2r_cvt',
            'in_chans':3, 'act_layer':nn.GELU, 'norm_layer':nn.LayerNorm, 'init':'trunc_norm',
            'spec':{
                'INIT': 'trunc_norm',
                'NUM_STAGES': 3,
                
                'ATTN_LAYER': [['Attention'], ['Attention', 'Attention_v5'], ['Attention']*10],
                'PATCH_SIZE': [7, 3, 3],
                'PATCH_STRIDE': [4, 2, 2],
                'PATCH_PADDING': [2, 1, 1],
                'DIM_EMBED': [64, 192, 384],
                'NUM_HEADS': [1, 3, 6],
                'DEPTH': [1, 2, 10],
                'MLP_RATIO': [4.0, 4.0, 4.0],
                'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
                'DROP_RATE': [0.0, 0.0, 0.0],
                'DROP_PATH_RATE': [0.0, 0.0, 0.1],
                'QKV_BIAS': [True, True, True],
                'CLS_TOKEN': [False, False, True],
                'POS_EMBED': [False, False, False],
                'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
                'KERNEL_QKV': [3, 3, 3],
                'PADDING_KV': [1, 1, 1],
                'STRIDE_KV': [2, 2, 2],
                'PADDING_Q': [1, 1, 1],
                'STRIDE_Q': [1, 1, 1]
            },
            'pretrain': '/data/udd3022/weights/cvt/CvT-13-224x224-IN-1k.pth'
        },
    
        #'proto_layer': {
            #'arch': 'fc',
            #'bias': True

            #'arch': 'mbo',
            #'in_features': 512*4,
            #'w_c': [0.7464113181504486, 3.620554021257009, 0.7137412766238266, 0.4484317018056678, 1.5320490119696863, 1.3127199902900837, 7.0551206784083496, 2.3436805894143777, 2.1670006010819476, 4.484967862326353, 4.343138239132617, 4.9402763503482925, 2.756594877023066, 43.831813576494426],
            #'w_A': 1.0/15.0,

            #'arch':'var_xproto_v2',
            #'prototype_shape':(14*3, 128, 1, 1), 
            #'prototype_activation_function':'linear', 
            #'add_on_layers_type':'regular',
            #'distance_type': 'cosine',

            #'arch':'xproto',
            #'prototype_shape':(14*3, 128, 1, 1), 
            #'prototype_activation_function':'linear', 
            #'add_on_layers_type':'regular',
            #'distance_type': 'cosine',

            #'arch': 'ppool',
            #'num_prototypes': 15,
            #'num_descriptive':10,
            #'proto_depth': 256,
            #'use_last_layer': True,
            #'prototype_activation_function': 'log',

            #'arch':'protop',
            #'img_size': 512,
            #'prototype_shape':(14*3, 512, 1, 1),
            #'init_weights':True,
            #'prototype_activation_function':'log',
            #'add_on_layers_type':'linear'
        #},

        'last_layer': {
            'arch': 'fc',
            'bias': True,
            'in_channels': 384
            
            #'arch':'class_wise_proto2logit',
            #'in_channels':14*3,

            #'arch': 'mbo',
            #'in_channels': 14*3,
            #'w_c': [0.7464113181504486, 3.620554021257009, 0.7137412766238266, 0.4484317018056678, 1.5320490119696863, 1.3127199902900837, 7.0551206784083496, 2.3436805894143777, 2.1670006010819476, 4.484967862326353, 4.343138239132617, 4.9402763503482925, 2.756594877023066, 43.831813576494426],
            #'w_A': 1.0/15.0,
        }
    },
    #'resume': None,


    # Optimizers
    #'warm_optim': {'method':'Adam', 'weight_decay':1e-3},
    #'warm_lrs': {'add_on_layers': 1e-3, 'prototype_vectors': 1e-3},
    #'warm_lr_schedule': {'method': 'CosineAnnealingLR', 'T_max':100},

    #'joint_optim': {'method':'Adam', 'weight_decay':1e-3},
    #'joint_lrs': {'features': 1e-4, 'add_on_layers': 1e-3, 'prototype_vectors': 1e-3, 'last_layer_lr':1e-3},
    #'joint_lr_schedule': {'method': 'CosineAnnealingLR', 'T_max':100},
    #'joint_lr_schedule': {'method': 'StepLR', 'step_size':5, 'gamma':0.1},
    
    'optim': {'method':'Adam', 'weight_decay':1e-4},
    'lr_schedule': {'method': 'CosineAnnealingLR', 'T_max':100},
    'lr': 1e-4,

    # Loss Functions
    'loss':[
        ({'method':'WeightedBalanceLoss', 'gamma':2.0, 'apply_sigmoid':True}, 1.0),
        #({'method':'BCEWithLogitsLoss'}, 1.0),
        #({'method':'hydra_vit_mlce_loss'}, 1.0),
        #({'method':'hydra_vit_cl_loss'}, 1.0),
        #({'method':'cluster_loss_v1'}, 0.5),
        #({'method':'separation_loss_v1'},  -0.5),
        #({'method':'last_layer_l1'}, 1e-4),
        #({'method':'last_layer_l1_with_pcid'}, 1e-4),
        #({'method':'orthogonal_loss_v2'}, 0.5),
        #({'method':'proto_kld'}, 0.00025),
        #({'method':'feat_kld'}, 0.00025)
    ],


    # Training schedule
    'start_epoch':0, 
    'epochs': 100,
    'warmup_epochs': 0,
    #'last_layer_iter_epoch':20,


    #'train_only_layers': [],
    #'clip_grad_norm': 1.0,

    # Best model selection criteria
    # track {dict} --> accuracy:1 | macro avg_{precision,recall,f1-score}:1 | weighted avg_{precision,recall,f1-score}:1 | <label_id>_{precision,recall,f1-score}
    # {<metric>:1|0} 1: higher is better, 0: lower is better
    'track': {'auc':1},

    # Logging frequencies
    'print_freq': 100, # [iter]
    'image_freq': 40, # [iter]
    'val_print_freq': 100, # [iter]
    #'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 5 # Model saving frequency [epoch]

}


def get():
    global opt
    return opt