

coefficients = {
    "distanceCoefficient": 1.0,
    "orientationCoefficient": 0.0,
    "velocityCoefficient": 0.0,
    "eventCoefficient": 0.0,
}

mit nur dem easy parcour im training mit fixed spawn konnte den easy parcour laufen

geht das auch mit random spawn?

kann es dann lernen das erste Tor zu finden?

# Distance Coefficient 1.0 Random Spawn

PPO_230

coefficients = {
    "distanceCoefficient": 1.0,
    "orientationCoefficient": 0.0,
    "velocityCoefficient": 0.0,
    "eventCoefficient": 0.0,
}

war nicht erfolgreich
easy_success_rate war immer 0.0


# Orientation und velocity coefficient

{'n_envs': 10, 'n_epochs': 5, 'batch_size': 64, 'n_steps': 128, 'coefficients': None, 'env_kwargs': {'spawn_point_random': True, 'single_goal': False, 'frame_stacking': 3, 'equalize': True, 'normalize_images': False, 'coefficients': {'distanceCoefficient': 0.0, 'orientationCoefficient': 1.0, 'velocityCoefficient': 10.0, 'eventCoefficient': 0.0}, 'mapType': 'randomEvalEasy'}}


# mit weiterem Sichtfeld und random spawn war es dann möglich ein paar erste Tore zu durchfahren

{'comment': 'With random spawn the first goal might not be visible, will try different camera input sizes (more width)', 'n_envs': 10, 'n_epochs': 5, 'batch_size': 64, 'n_steps': 128, 'coefficients': None, 'env_kwargs': {'spawn_point_random': True, 'single_goal': False, 'frame_stacking': 3, 'equalize': True, 'normalize_images': False, 'coefficients': {'distanceCoefficient': 0.0, 'orientationCoefficient': 1.0, 'velocityCoefficient': 10.0, 'eventCoefficient': 0.0}, 'mapType': 'randomEvalEasy', 'width': 500, 'height': 168}}
