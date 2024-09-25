from utils import *

from datetime import datetime

class setting_config:
    """
    the config of training setting.
    """
    network = 'LightConvSeg' 
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [32, 64, 72, 96, 128],
        # 'c_list': [32, 48, 64, 72, 96, 128], 
        'pretrained_path' : 'encoder_pretrain/Crack500_encoder_pretrained.pth'
    }

    test_weights = ''

    datasets = 'CRACK_DATASET'
    if datasets == 'CRACK_DATASET':
        data_path = ''
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8
    epochs = 300

    work_dir = 'results/' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 200
    val_interval = 10
    save_interval = 100
    threshold = 0.5

    opt = 'AdamW'
    lr = 0.001 # default: 1e-3 – learning rate
    betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
    eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
    weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
    amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 

    sch = 'CosineAnnealingLR'
    T_max = 50 # – Maximum number of iterations. Cosine function period.
    eta_min = 0.00001 # – Minimum learning rate. Default: 0.
    last_epoch = -1 # – The index of last epoch. Default: -1.  