import os

def get_from_env_or_error(env_key: str, error_message: str, error_class = ValueError):
    env_value = os.getenv(env_key)
    if not env_value:
        raise error_class(error_message.format(env_key=env_key))
    
    return env_value