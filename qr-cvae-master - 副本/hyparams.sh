# python run.py --multirun \
#     model_params=sim_cqrvae \
#     exp_params=sim_c \
#     data_params=sim_c \
#     trainer_params=sim_trainer \
#     exp_params.kld_weight=0.8,0.5 \
#     exp_params.LR=0.001,0.0005,0.0001 \
#     exp_params.weight_decay=0.0,0.01,0.001 \
#     exp_params.scheduler_gamma=null,0.9,0.95 \
#     exp_params.manual_seed=42,43,44

###### after Hyperparameter Tuning
#1
python run.py --multirun \
    model_params=sim_cqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python evaluate_cond.py --multirun \
    model_params=sim_cqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8  \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

#2
python run.py --multirun \
    model_params=sim_cvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python evaluate_cond.py --multirun \
    model_params=sim_cvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44



python run.py --multirun \
    model_params=simqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44
#3
python run.py --multirun \
    model_params=simvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

#4


# evaluation
#1
python evaluate_cond.py --multirun \
    model_params=simqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44
#2

#3
python evaluate_cond.py --multirun \
    model_params=simvae \
    exp_params=sim_c \
    data_params=sim_c \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

#4


