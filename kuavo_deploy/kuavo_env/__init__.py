from gymnasium.envs.registration import register
register(
    id='Kuavo-Sim',
    entry_point='kuavo_deploy.kuavo_env.kuavo_sim_env.KuavoSimEnv:KuavoSimEnv',
)

register(
    id='Kuavo-Real',
    entry_point='kuavo_deploy.kuavo_env.kuavo_real_env.KuavoRealEnv:KuavoRealEnv',
)