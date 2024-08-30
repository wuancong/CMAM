from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.M1_FEAT_REDUCE = CN()
_C.MODEL.M1_FEAT_REDUCE.USE = False
_C.MODEL.M1_FEAT_REDUCE.DIM = 1024

_C.MODEL.BASE = "BASE"
_C.MODEL.META_ARCHITECTURE = 'NEWBASE'
_C.MODEL.OPEN_LAYERS = ['']
_C.MODEL.OUTCUDA = True
_C.MODEL.TRANSFORMER = 'VITB'

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.MAM = False
_C.MODEL.MAM_CNT = 'BASE'
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = 50
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# Mini-batch split of Ghost BN
_C.MODEL.BACKBONE.NORM_SPLIT = 1
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
_C.MODEL.BACKBONE.WITH_CICONV = False
_C.MODEL.BACKBONE.CICONV_INVARIANT = 'N' #['E', 'H', 'N', 'W', 'C']
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "BNneckHead"

# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Mini-batch split of Ghost BN
_C.MODEL.HEADS.NORM_SPLIT = 1
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = -1
# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 512
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcface" or "circle"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128


# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

_C.MODEL.LOSSES.CENTER = CN()
_C.MODEL.LOSSES.CENTER.ENABLED = False
_C.MODEL.LOSSES.CENTER.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0
_C.MODEL.LOSSES.TRI.SQUARED = False

_C.MODEL.LOSSES.TRIMOD = CN()
_C.MODEL.LOSSES.TRIMOD.SCALE = 0.0
_C.MODEL.LOSSES.TRIMOD.MARGIN = 0.3
_C.MODEL.LOSSES.TRIMOD.NORM_FEAT = False
_C.MODEL.LOSSES.TRIMOD.DIV = 1.0
_C.MODEL.LOSSES.TRIMOD.HARD_MINING = True

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.ALPHA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0


#CC Loss optiions
_C.MODEL.LOSSES.CC = CN()
_C.MODEL.LOSSES.CC.MARGIN = 0.3
_C.MODEL.LOSSES.CC.ALPHA = 128
_C.MODEL.LOSSES.CC.SCALE = 0.0

#PRED Loss optiions
_C.MODEL.LOSSES.PRED = CN()
_C.MODEL.LOSSES.PRED.SCALE = 0.1
_C.MODEL.LOSSES.PRED.CE_SCALE = 0.0


_C.MODEL.SEPCHN = CN()
_C.MODEL.SEPCHN.RGB = False


#Transformer Parameter
_C.MODEL.TRANSFORMER_TYPE='vit_base_patch16_224_TransReID'
_C.MODEL.STRIDE_SIZE=[12, 12]
_C.MODEL.ATT_DROP_RATE=0.0
_C.MODEL.DROP_OUT=0.0
_C.MODEL.DROP_PATH=0.1
_C.MODEL.LAST_STRIDE=1
_C.MODEL.PRETRAIN_PATH = '/home/chengzhi/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
_C.MODEL.PATCH_DROP = 0.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

_C.MODEL.LOSSES.USE_OUT5 = False
_C.MODEL.LOSSES.USE_OUT4 = False

#MSE Loss
_C.MODEL.LOSSES.MSE = CN()
_C.MODEL.LOSSES.MSE.SCALE = 0.0
_C.MODEL.LOSSES.MSE.ADD_LOW = False

#MSE Loss
_C.MODEL.LOSSES.RGB_REPEAT_MSE = CN()
_C.MODEL.LOSSES.RGB_REPEAT_MSE.SCALE = 0.0

#AMSE
_C.MODEL.LOSSES.AMSE = CN()
_C.MODEL.LOSSES.AMSE.SCALE = 0.0
_C.MODEL.LOSSES.AMSE.ADD_LOW = True

#SIM
_C.MODEL.LOSSES.SIM = CN()
_C.MODEL.LOSSES.SIM.SCALE = 0.0
_C.MODEL.LOSSES.SIM.MEMORY = False
_C.MODEL.LOSSES.SIM.END = 1
_C.MODEL.LOSSES.SIM.LOGMATRIX = False
_C.MODEL.LOSSES.SIM.METHOD = 'BASE'
_C.MODEL.LOSSES.SIM.USE = [1,1,1,1]
_C.MODEL.LOSSES.SIM.PROJ = False
_C.MODEL.LOSSES.CROSS_SIM = CN()
_C.MODEL.LOSSES.CROSS_SIM.SCALE = 0.0


_C.MODEL.LOSSES.DIVERGE = CN()
_C.MODEL.LOSSES.DIVERGE.SCALE = 0.0

_C.MODEL.LOSSES.FEA = CN()
_C.MODEL.LOSSES.FEA.SCALE = 0.0

_C.MODEL.LOSSES.ORTH = CN()
_C.MODEL.LOSSES.ORTH.CE = False
_C.MODEL.LOSSES.ORTH.CE_TARGET = "PID"
_C.MODEL.LOSSES.ORTH.CE_SCALE = 0.0
_C.MODEL.LOSSES.ORTH.TRI = False
_C.MODEL.LOSSES.ORTH.TRI_SCALE = 0.0

_C.MODEL.LOSSES.ORTH.KL = False
_C.MODEL.LOSSES.ORTH.KL_TARGET = 0
_C.MODEL.LOSSES.ORTH.KL_SCALE = 0.0

_C.MODEL.LOSSES.ORTH.DETACH = False

_C.MODEL.LOSSES.ORTH.ALLFEA = False
_C.MODEL.LOSSES.ORTH.FEATORTH = False
_C.MODEL.LOSSES.ORTH.ADDMAP = False
_C.MODEL.LOSSES.ORTH.POOL = False
_C.MODEL.LOSSES.CAMTARGET = False



_C.MODEL.LOSSES.MM = CN()
_C.MODEL.LOSSES.MM.SCALE = 0.0
_C.MODEL.LOSSES.MM.SIDWEI = 0.5
_C.MODEL.LOSSES.MM.KLWEI = 2.0
_C.MODEL.LOSSES.MM.LAMBDA = 0.2

_C.MODEL.LOSSES.MMD = CN()
_C.MODEL.LOSSES.MMD.SCALE = 0.0
_C.MODEL.LOSSES.MMD.LAMBDA = 0.8
_C.MODEL.LOSSES.MMD.INTER = 1.0

_C.MODEL.LOSSES.KL = CN()
_C.MODEL.LOSSES.KL.SCALE = 0.0
_C.MODEL.LOSSES.KL.BEGIN = 0
_C.MODEL.LOSSES.KL.LOGIT = False


_C.MODEL.LOSSES.CA = CN()
_C.MODEL.LOSSES.CA.SCALE = 0.0
_C.MODEL.LOSSES.CA.META = False
_C.MODEL.LOSSES.CA.META_LR = 0.1
_C.MODEL.LOSSES.CA.MEMORY = False


_C.MODEL.LOSSES.ALIGN = CN()
_C.MODEL.LOSSES.ALIGN.SCALE = 0.0
_C.MODEL.LOSSES.ALIGN.LOW_USE = False
_C.MODEL.LOSSES.ALIGN.LOW_AVG = False
_C.MODEL.LOSSES.ALIGN.HIGH_USE = False
_C.MODEL.LOSSES.ALIGN.NORM = False
_C.MODEL.LOSSES.ALIGN.SQUARE = False
_C.MODEL.LOSSES.ALIGN.L2 = False

_C.MODEL.LOSSES.SUB = CN()
_C.MODEL.LOSSES.SUB.SCALE = 0.0
_C.MODEL.LOSSES.SUB.CE_SCALE = 0.0
_C.MODEL.LOSSES.SUB.FUSE_SCALE = 0.0

_C.MODEL.LOSSES.M6 = CN()
_C.MODEL.LOSSES.M6.SCALE = 1.0


_C.MODEL.LOSSES.PIXEL = CN()
_C.MODEL.LOSSES.PIXEL.SCALE = 0.0


_C.MODEL.LOSSES.DIS = CN()
_C.MODEL.LOSSES.DIS.RECSCALE = 1.0
_C.MODEL.LOSSES.DIS.CESCALE = 0.33333
_C.MODEL.LOSSES.DIS.ORTHSCALE = 1.0


_C.MODEL.LOSSES.SUB.ONLY_MEAN = False

_C.MODEL.LOSSES.CROSS = CN()

_C.MODEL.LOSSES.CROSS.LOGIT = CN()
_C.MODEL.LOSSES.CROSS.LOGIT.SCALE = 0.0


# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]
#


_C.MODEL.RDROP = CN()
_C.MODEL.RDROP.SCALE = 0.0
_C.MODEL.RDROP.P = 0.0
_C.MODEL.RDROP.BEGIN = 0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.CAJ = False
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5


_C.INPUT.ORDER = True
# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.DO_CJ = False
_C.INPUT.CJ = CN()

_C.INPUT.CJ.BRI = 0.2
_C.INPUT.CJ.CON = 0.15
_C.INPUT.CJ.SAT = 0.0
_C.INPUT.CJ.HUE = 0.0

_C.INPUT.HP = CN()
_C.INPUT.HP.ENABLED = False
_C.INPUT.HP.BRI = 0.2
_C.INPUT.HP.CON = 0.15
_C.INPUT.HP.SAT = 0.0
_C.INPUT.HP.HUE = 0.0
_C.INPUT.HP.NOISE = 10
_C.INPUT.HP.PTHFN = ''
_C.INPUT.HP.PROB = 0.2



#(brightness=0.2, contrast=0.15, saturation=0, hue=0
# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.CREA = False
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [0.596*255, 0.558*255, 0.497*255]  # [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5

_C.INPUT.CHANNEL = CN()

_C.INPUT.CHANNEL.SHUFFLE = CN()
_C.INPUT.CHANNEL.SHUFFLE.ENABLED = False
_C.INPUT.CHANNEL.SHUFFLE.NAIVE = False
_C.INPUT.CHANNEL.SHUFFLE.HALF_REPLACE = False
_C.INPUT.CHANNEL.SHUFFLE.TOTAL_REPLACE = False
_C.INPUT.CHANNEL.SHUFFLE.TOTAL_SEP = False
_C.INPUT.CHANNEL.SHUFFLE.TOTAL_GRAY = False
_C.INPUT.CHANNEL.SHUFFLE.TOTAL_WEIGHTED = False
_C.INPUT.CHANNEL.SHUFFLE.ZERO_SEP = False
_C.INPUT.CHANNEL.WEIGHTED = False
_C.INPUT.CHANNEL.GRAY = False
_C.INPUT.CHANNEL.SINGLE = False
_C.INPUT.CHANNEL.ONLYONE = False
_C.INPUT.CHANNEL.ORDER = -1


_C.INPUT.CHANNEL.TOTAL = CN()
_C.INPUT.CHANNEL.TOTAL.ENABLED = False
_C.INPUT.CHANNEL.TOTAL.GRAY_USE = False
_C.INPUT.CHANNEL.TOTAL.WEI_USE = False
_C.INPUT.CHANNEL.TOTAL.HALF_REPLACE = False

_C.INPUT.CHANNEL.AUTOSTA = False

_C.INPUT.REPEAT_THIRD_MIXUP = False
_C.INPUT.REPEAT_THIRD_MIXUP_WEIGHT = 0.5
_C.INPUT.REPEAT_THIRD_MIXUP_PROB = 1.0
_C.INPUT.REPEAT_THIRD_CONCAT = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False
_C.DATASETS.REGDB_SPLIT_INDEX = 1


_C.DATASETS.NAME2 = "sysu-mm01"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = False
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.IR_SAMPLER = False
_C.DATALOADER.IRCONTRAST_SAMPLER = False


_C.ADV = CN()
_C.ADV.K = 5
_C.ADV.NOISE = ""

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.USE_SURGERY = False
_C.SOLVER.SURGERY_METHOD = 'sign'
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.
_C.SOLVER.HEADS_LR_FACTOR = 1.
_C.SOLVER.BODY_LR_FACTOR = 1.
_C.SOLVER.IR_LR_FACTOR = 1000

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Multi-step learning rate options
_C.SOLVER.SCHED = "WarmupMultiStepLR"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

# Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 100
_C.SOLVER.ETA_MIN_LR = 3e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.FREEZE_ITERS = 0

# SWA options
_C.SOLVER.SWA = CN()
_C.SOLVER.SWA.ENABLED = False
_C.SOLVER.SWA.ITER = 0
_C.SOLVER.SWA.PERIOD = 10
_C.SOLVER.SWA.LR_FACTOR = 10.
_C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
_C.SOLVER.SWA.LR_SCHED = False

_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.LOG_PERIOD = 30
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.FLIP = False

_C.TEST.EVAL_PERIOD = 50
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.METRIC = "cosine"
_C.TEST.MODALITY = 0 # 0: do nothing, 1: save rgb-rgb ranking list, 2: save rgb-ir ranking list for mm01 only
_C.TEST.SAVE_FEATURE = False

# Average query expansion
_C.TEST.AQE = CN()
_C.TEST.AQE.ENABLED = False
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CN()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300
_C.TEST.EVASETTING = ''

#LTM
_C.LTM = CN( )
_C.LTM.ENABLED = False
_C.LTM.TESTLOW = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"
_C.SAVE_PRE = ''

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

_C.NOISE_SCALE = 1.0

_C.AUXI_LOSS = ('MEAN',)
_C.AUXI_WEI = (0.0,)

_C.COPYBRANCH = False

_C.META = CN()
_C.META.LR = 0.01
_C.META.SIM_LR = 0.0
_C.META.USE = False

_C.MUTUAL = CN()
_C.MUTUAL.SAMEMODEL = False
_C.MUTUAL.DROPUOUT = 0.0
_C.MUTUAL.SECOND = CN()
_C.MUTUAL.FIRST = CN()
_C.MUTUAL.FIRST.INPUT = 'RGB'
_C.MUTUAL.FIRST.INPUT_SAMPLE = 'ORIGIN+GEN' # ORIGIN OR GEN OR ORIGIN+GEN
_C.MUTUAL.FIRST.JOINT = False
_C.MUTUAL.SOFT = False
_C.MUTUAL.SECOND.ENABLED = False
_C.MUTUAL.SECOND.INPUT = 'RGB'
_C.MUTUAL.SECOND.INPUT_SAMPLE = 'GEN' # ORIGIN OR GEN OR ORIGIN+GEN
_C.MUTUAL.SECOND.IMAGES = 1
_C.MUTUAL.SECOND.COPYFIRST = False
_C.MUTUAL.STOPGRAD = False
_C.MUTUAL.USE2TEST = False
_C.MUTUAL.ONLY2TEST = False
_C.MUTUAL.LOSSES = ("CrossEntropyLoss",)
_C.MUTUAL.FEATADJUST = False
_C.MUTUAL.STOPGRAD = False
_C.MUTUAL.WEIGHTDECAY = False
_C.MUTUAL.BEGIN = 0
_C.MUTUAL.TYPE = 'SYMMETRIC'
_C.MUTUAL.USE_SPECIFIC_HEAD = False

_C.MUTUAL.MULTI = CN()
#For CNN
_C.MUTUAL.MULTI.POSUSE = [0,0,0,1]
_C.MUTUAL.MULTI.POSDIM = [256,512,1024,2048]
#For Transformer
_C.MUTUAL.MULTI.TRANSUSE = [0,0,0,0]
_C.MUTUAL.MULTI.TRANSHEADS = [3,3,3,3]
_C.MUTUAL.MULTI.TRANSMETHOD = 'BASE'
_C.MUTUAL.MULTI.SWIN = CN()

_C.MUTUAL.MULTI.SWIN.NUM_HEADS = 8
_C.MUTUAL.MULTI.SWIN.WIN_SIZE = 4
_C.MUTUAL.MULTI.SWIN.POS_EMBED = False

_C.MUTUAL.WITHNL = False

_C.MUTUAL.MODEL = CN()
_C.MUTUAL.MODEL.USE = -1

_C.MUTUAL.PRED = CN()
_C.MUTUAL.PRED.ENABLED = False
_C.MUTUAL.PRED.SCALE = 0.0
_C.MUTUAL.PRED.PERCEP = 0.0
_C.MUTUAL.PRED.STOP_GRAD = False
_C.MUTUAL.PRED.LOSS = 'MSE'
_C.MUTUAL.TYPE = 'SYMMETRIC'

_C.DISTILL = CN()
_C.DISTILL.DIRECTION = 2

_C.CROSS = CN()
_C.CROSS.LOSSES = ("CrossEntropyLoss",)
_C.STATE = 'RGB'

_C.UL = CN()
_C.UL.SUPER = False
_C.UL.SEMI = False
_C.UL.SEMI_RATIO = 0.0 # the proportion of labeled IDs

_C.UL.CLUSTER = CN()
_C.UL.CLUSTER.EQDIST = True
_C.UL.CLUSTER.NUM = 200
_C.UL.CLUSTER.METHOD = 'CROSS_KMEANS'
_C.UL.CLUSTER.USE_THIRD_MODALITY = False
_C.UL.CLUSTER.CONNECT = 'BGM'
_C.UL.CLUSTER.STOP = -1

_C.UL.CLUSTER.BGM = CN()
_C.UL.CLUSTER.BGM.CIRCLE = False
_C.UL.CLUSTER.BGM.GREEDY = False

_C.UL.CLUSTER.FEASEP = False
_C.UL.CLUSTER.AQE = False
_C.UL.CLUSTER.AQE_AJ = False
_C.UL.CLUSTER.TIMES = 500
_C.UL.CLUSTER.REFER = ''
_C.UL.CLUSTER.ADD_REFER = False
_C.UL.CLUSTER.MEMORY = 'CM'
_C.UL.CLUSTER.MEMORY_TYPE = 'SHARE' # OPTIONS: SHARE MODALITY_SPECIFIC
_C.UL.CLUSTER.MEMORY_TEMP = 0.05

#label smooth
_C.UL.CLUSTER.LSCE = False
_C.UL.CLUSTER.MEMLR = 1e-4
_C.UL.CLUSTER.USE_HARD = False
_C.UL.CLUSTER.MOMEMTUM = 0.2
_C.UL.CLUSTER.PLFEA = -1
_C.UL.CLUSTER.TRIUSE = False
_C.UL.CLUSTER.SAMPLER = 'BASE'
_C.UL.CLUSTER.CENTER = 'BASE'
_C.UL.CLUSTER.INIT = ''
_C.UL.CLUSTER.ALTER_SPECIFIC_SHARE = False
_C.UL.CLUSTER.ALTER_DBSCAN_KMEANS = False
_C.UL.CLUSTER.USE_SINGLE_MODEL_INTRA_CLUSTER = False


_C.UL.CLUSTER.CATCLS = False
_C.UL.PURE = CN()
_C.UL.PURE.LOSSES = ("CrossEntropyLoss",)


_C.IS = 'MGN'
_C.TRANSFER = CN()
_C.TRANSFER.MODEL = "RGB"