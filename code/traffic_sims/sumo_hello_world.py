from flow.networks.ring import RingNetwork

from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import NetParams
from flow.core.params import InitialConfig

from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams

from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import EnvParams

from flow.core.experiment import Experiment


vehicles = VehicleParams()

vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=32)


print(ADDITIONAL_NET_PARAMS)


net_params = NetParams(additional_params={
    "radius": 80,
    'length': 330,
    'lanes': 3,
    'speed_limit': 10,
    'resolution': 40
})


initial_config = InitialConfig(spacing="uniform", perturbation=1)

traffic_lights = TrafficLightParams()

sumo_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)


# create the network object
network = RingNetwork(name="ring_example",
                      vehicles=vehicles,
                      net_params=net_params,
                      initial_config=initial_config,
                      traffic_lights=traffic_lights)

# create the environment object
env = AccelEnv(env_params, sumo_params, network)

# create the experiment object
exp = Experiment(env)

# run the experiment for a set number of rollouts / time steps
_ = exp.run(1, 3000, convert_to_csv=True)


print(ADDITIONAL_ENV_PARAMS)
