

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



# bilder aus Memory zeigen, dass das Memory nicht sehr weit zurueckgeht
vielleicht reicht n=3 nicht aus 



# train with random spawn and only event reward possible?

no, not possible

# train with random spawn and event + distance reward possible?

not clear, it showed some learning

eval_easy/rate_first_goal erreicht 0.3


# all maps train with fixed spawn and event + distance reward possible?

Agent lernt immer geradeaus zu fahren
(bei fixed spawn bedeutet diese Strategie perfekte Ergebnisse für den easy und medium parcour)
(gar kein Erfolg für den hard parcour)

vielleicht brauchen wir einen weniger random Spawn im training oder ein training mit purem distance reward

{'comment': 'all maps train with fixed spawn and event + distance reward possible?', 'n_envs': 100, 'num_evals_per_difficulty': 100, 'n_epochs': 5, 'log_interval': 1, 'batch_size': 64, 'n_steps': 64, 'copy_model_from': 'models_and_dumps/best_model_episode_8', 'env_kwargs': {'spawn_point_random': False, 'single_goal': False, 'frame_stacking': 3, 'image_preprocessing': {'grayscale': True, 'equalize': True, 'contrast_increase': 'TODO', 'normalize_images': False}, 'coefficients': {'distanceCoefficient': 0.5, 'orientationCoefficient': 0.0, 'velocityCoefficient': 0.0, 'eventCoefficient': 1.0}, 'mapType': 'random', 'width': 500, 'height': 168}}

