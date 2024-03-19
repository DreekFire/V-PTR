
# Online RL

## SAC
```bash
python train_online.py --env_name=HalfCheetah-v2
```

## DrQ
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online_pixels.py --env_name=cheetah-run-v0
```

# Offline RL

## BC (TanhNormal)
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=tanh_normal --env_name=halfcheetah-expert-v2
```

## BC (Autoregressive Policy)
```bash
python train_offline.py --config=configs/offline_config.py:bc --config.model_config.distr=ar --env_name=halfcheetah-expert-v2
```

## IQL
```bash
python train_offline.py --config=configs/offline_config.py:iql --env_name=antmaze-large-play-v2 --eval_interval=100000 --eval_episodes=100
```

## For roboverse
```bash
python train_offline.py --config.distr=ar --env_name=halfcheetah-expert-v2
```

## For roboverse
```bash
python train_offline_pixels.py  
```