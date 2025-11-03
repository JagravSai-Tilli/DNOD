num_classes = 7
num_workers = 8 
dataset_file = "SAR_DET100K"
data_path = "../data/SAR_DET100K"

weight_decay = 0.0001
epochs = 56
save_checkpoint_interval = 1
clip_max_norm = 0.1
# lr
lr = 0.0001
lr_backbone = 0.00001  
lr_drop = 52
#o2m
use_aux_ffn = True
o2m_matcher_threshold = 0.4
o2m_matcher_k = 6
use_indices_merge = False
o2m_cls_loss_coef = 2
o2m_bbox_loss_coef = 5
o2m_giou_loss_coef = 2

# backbone
backbone = "resnet50"
dilation = 0
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20

#resnet50 backbone
hidden_dim = 256
enc_layers = 3
dec_layers = 3
pre_norm = False
dim_feedforward = 2048
dropout = 0.0
nheads = 8
num_queries = 1200  
query_dim = 4  
num_feature_levels = 3
list_backbone_levels = [-1, -2, -3]
dec_n_points = 6
num_select = 1200  
transformer_activation = "relu"
masks = False 
# loss
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
matcher_type = "HungarianMatcher"
nms_iou_threshold = 0.8

decoder_module_seq = ["sa", "ca", "ffn"]


