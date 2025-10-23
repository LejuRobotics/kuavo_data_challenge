from gymnasium.envs.registration import register
from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.kuavo_env.KuavoRealEnv import KuavoRealEnv
register(
    id='Kuavo-Sim',
    entry_point='kuavo_deploy.kuavo_env.KuavoSimEnv:KuavoSimEnv',
)

register(
    id='Kuavo-Real',
    entry_point='kuavo_deploy.kuavo_env.KuavoRealEnv:KuavoRealEnv',
)
__all__ = ["KuavoSimEnv","KuavoRealEnv"]