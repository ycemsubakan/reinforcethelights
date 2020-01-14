import logging
import numpy as np
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

# from flow.core.experiment import Experiment

import pdb

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

sumo_params = SumoParams(sim_step=0.1, render=False, emission_path='data')

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)


# create the network object
network = RingNetwork(name="ring_example",
                      vehicles=vehicles,
                      net_params=net_params,
                      initial_config=initial_config,
                      traffic_lights=traffic_lights)

# create the environment object
env = AccelEnv(env_params, sumo_params, network)
# num_steps = env.env_params.horizon
num_steps = 10
print('num steps {}'.format(num_steps))

NUM_RUNS = 10


def rl_actions(*_):
    return None


info_dict = {}
rets = []
mean_rets = []
ret_lists = []
vels = []
mean_vels = []
std_vels = []
outflows = []
for i in range(NUM_RUNS):
    vel = np.zeros(num_steps)
    logging.info("Iter #" + str(i))
    ret = 0
    ret_list = []
    state = env.reset()
    for j in range(num_steps):
        state, reward, done, _ = env.step(rl_actions(state))
        vel[j] = np.mean(
            env.k.vehicle.get_speed(env.k.vehicle.get_ids()))
        ret += reward
        ret_list.append(reward)

        if done:
            break
    rets.append(ret)
    vels.append(vel)
    mean_rets.append(np.mean(ret_list))
    ret_lists.append(ret_list)
    mean_vels.append(np.mean(vel))
    std_vels.append(np.std(vel))
    outflows.append(env.k.vehicle.get_outflow_rate(int(500)))
    print("Round {0}, return: {1}".format(i, ret))

info_dict["returns"] = rets
info_dict["velocities"] = vels
info_dict["mean_returns"] = mean_rets
info_dict["per_step_returns"] = ret_lists
info_dict["mean_outflows"] = np.mean(outflows)

print("Average, std return: {}, {}".format(
    np.mean(rets), np.std(rets)))
print("Average, std speed: {}, {}".format(
    np.mean(mean_vels), np.std(mean_vels)))
env.terminate()

# create the experiment object
# exp = Experiment(env)

# run the experiment for a set number of rollouts / time steps
# _ = exp.run(1, 3000, convert_to_csv=True)


# print(ADDITIONAL_ENV_PARAMS)
