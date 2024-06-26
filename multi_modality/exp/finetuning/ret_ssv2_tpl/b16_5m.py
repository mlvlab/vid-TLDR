from configs.data import *
from configs.model import *

# ========================= data ==========================
del available_corpus

train_file = [
    f"{anno_root_downstream}/ssv2_ret_template_train.json",
    "your_sthv2_path",
    "video",
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/ssv2_ret_template_val_small.json",
        "your_sthv2_path",
        "video",
    ],
)
test_types = ["val"]
num_workers = 6

stop_key =  "val/" # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = False

has_multi_vision_gt = True

# ========================= input ==========================
num_frames = 12
num_frames_test = 12
batch_size = 32
max_txt_l = 25

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
text_enc = "bert"
model = dict(
    model_cls="UMT",
    vision_encoder=dict(
        # backbone
        name="vit_b16",
        img_size=224, 
        patch_size=16, 
        d_model=768,
        encoder_embed_dim=768, 
        encoder_depth=12,
        encoder_num_heads=12, 
        drop_path_rate=0.1, 
        num_frames="${num_frames}",
        tubelet_size=1,
        use_checkpoint=True,
        checkpoint_num=12,
        clip_decoder_embed_dim=768,
        clip_output_dim=512,
        clip_return_layer=0,
        clip_student_return_interval=1,
        pretrained="your_model_path/b16_ptk710_f8_res224.pth",
        # clip teacher
        clip_teacher="none",
        clip_img_size=224,
        clip_return_interval=1,
        # mask
        video_mask_type="attention",
        video_mask_ratio=0.,
        video_double_mask_ratio=0.,
        image_mask_type="attention",
        image_mask_ratio=0.,
        image_double_mask_ratio=0.,
        # for ret
        keep_temporal=False,
        vidTLDR_use   = False,
        vidTLDR_r     = None,
        
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        mlm=0.0, 
        vtm=1.0, 
        uta=0.0,
    ),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    uta_norm_type="l2",
    uta_loss_type="l2",
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=1)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="umt_ret_ssv2",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?


