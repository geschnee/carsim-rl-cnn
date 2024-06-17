

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

{'comment': 'all maps train with fixed spawn and event + distance reward possible?', 'n_envs': 100, 'num_evals_per_difficulty': 100, 'n_epochs': 5, 'log_interval': 1, 'batch_size': 64, 'n_steps': 64, 'copy_model_from': 'models_and_dumps/best_model_episode_8', 'env_kwargs': {'spawn_point_random': False, 'single_goal': False, 'frame_stacking': 3, 'image_preprocessing': {'grayscale': True, 'equalize': True, 'normalize_images': False}, 'coefficients': {'distanceCoefficient': 0.5, 'orientationCoefficient': 0.0, 'velocityCoefficient': 0.0, 'eventCoefficient': 1.0}, 'mapType': 'random', 'width': 500, 'height': 168}}




# train on easyMap with less random Orientation fixed spawn pos

The training achieves very good performance for easy parcour even with the less random orientation spawn.
(less random means only +-15 Degrees orientation with fixed spawn pos)

see run 2024-01-23\22-37-30\tmp\PPO_1 on Desktop PC


comment: "reason for low fps is mainly n_envs, we now try less environments and less evals"
n_envs: 10
num_evals_per_difficulty: 20
n_epochs: 5 # amount of training passes over the replay buffer per timestep
log_interval: 5
batch_size: 64


n_steps: 64 # amount of steps to collect per collect_rollouts per environment
# Tensorboard shows the mean_episode_length is below 3
# we can reduce the n_steps to make the collection time shorter while still collecting a lot of games

# maybe the fps is too low, the agent does not have many steps to do and cannot react quick enough to learn a better policy than full on ahead

copy_model_from: models_and_dumps/best_model_episode_70
# copy_model_from can be False or a string (filename without .zip suffix)

env_kwargs:
  spawn_point: OrientationRandom
  # spawn_point can be Fixed, OrientationRandom, OrientationVeryRandom and FullyRandom
  single_goal: False
  frame_stacking: 3
  image_preprocessing:
    grayscale: True
    equalize: True
    # vielleicht eine art Mode variable hier reinpacken contrast increase s equalize
    normalize_images: False
  coefficients:
    distanceCoefficient: 0.5
    orientationCoefficient: 0.0
    velocityCoefficient: 0.0
    eventCoefficient: 1.0
  trainingMapType: randomEvalEasy
  width: 500 #336
  height: 168




# differentialJetBot SpawnPoint Very Random funktioniert sehr gut, Easy und MEdium wird mit fast 100% gelöst

Run 26.01.2024 15-16-49 auf Desktop PC

comment: "now trying again with fixed spawn pos but fully 45 Degree random spawn orientation, can it still reach success rate of 1 for easy parcour?"

n_envs: 10
num_evals_per_difficulty: 20
n_epochs: 5
log_interval: 5
batch_size: 64

n_steps: 64 

copy_model_from: False 

env_kwargs:
  jetbot: DifferentialJetBot
  spawn_point: OrientationVeryRandom
  single_goal: False
  frame_stacking: 3
  image_preprocessing:
    grayscale: True
    equalize: True
    normalize_images: False
  coefficients:
    distanceCoefficient: 0.5
    orientationCoefficient: 0.0
    velocityCoefficient: 0.0
    eventCoefficient: 1.0
  trainingMapType: randomEval
  width: 500 #336
  height: 168

# es sieht aus als bräuchte der differential JetBot den orientation reward oder so

der Differential Jetbot dreht sich oft falschrum


