from gymnasium.envs.registration import register
register(
    id='Kuavo-Sim',
    entry_point='kuavo_deploy.kuavo_env.kuavo_sim_env.KuavoSimEnv:KuavoSimEnv',
    max_episode_steps=150,
)

register(
    id='Kuavo-Real',
    entry_point='kuavo_deploy.kuavo_env.kuavo_real_env.KuavoRealEnv:KuavoRealEnv',
    max_episode_steps=150,
)