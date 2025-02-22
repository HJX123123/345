
python run.py --multirun \
    model_params=sim_cqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.01 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python run.py --multirun \
    model_params=sim_cvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.01 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python run.py --multirun \
    model_params=simqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=null \
    exp_params.manual_seed=42,43,44

python run.py --multirun \
    model_params=simvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=null \
    exp_params.manual_seed=42,43,44

# Eval
python evaluate_cond.py --multirun \
    model_params=simvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=null \
    exp_params.manual_seed=42,43,44

python evaluate_cond.py --multirun \
    model_params=sim_cvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.01 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python evaluate_cond.py --multirun \
    model_params=sim_cqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.01 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=0.95 \
    exp_params.manual_seed=42,43,44

python evaluate_cond.py --multirun \
    model_params=simqrvae \
    exp_params=sim_c \
    data_params=sim_c \
    data_params.name=sim_cond_v2 \
    trainer_params=sim_trainer \
    exp_params.kld_weight=0.8 \
    exp_params.LR=0.001 \
    exp_params.weight_decay=0.0 \
    exp_params.scheduler_gamma=null \
    exp_params.manual_seed=42,43,44
