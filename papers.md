# Things we read for the Traffic Lights Problem

* https://arxiv.org/pdf/1710.05465.pdf

## Deep Reinforcement Learning for Traffic Light Control in Vehicular Networks

### 1. Intro

The policeman in the intersection is replaced by a reinforcement learning algorithm. The policeman is assumed to follow an Markov Decision Process. If I am understanding correctly,  the overall congestion in the streets is represented as a matrix, and somehow we go from this matrix to a set of rewards, which is a function of cumulative waiting times. 

In this paper, the environment is simulated through a simulator called `SUMO`. 

### 2. Literature Review 

### 3. Model and Problem Statement 

#### Some Definitions
	
* Traffic lights are used to control the traffic flow. (yes, duh) 

* Traffic light gathers road traffic information via a vehicular network. Traffic light somehow processes the data to obtain road traffic's state and reward. 

* Then there is a decision making process: depending on the state, the traffic light somehow chooses an action which aims to maximize a reward function. I don't understand how/why but the road traffic is directly linked to a state and a reward. 

* At a given intersection with multiple directions, one might need to have multiple traffic lights. So the action space is larger. 

* Traffic light status : an allowable color configuration? 

* the duration for which one uses a status is called a phase. the (possible?) number of phases is the number of legal statuses. (in the paper they talk about number of phases, but the duration is a continuous number ?) 

* phases cyclically change in a fixed sequence to guide vehicles to pass the intersection. if we have a circular phase structure, we call that a cycle. 

* the sequence of phases in a cycle is fixed, but the duration of each phase is adaptive.  

#### The general idea is to maximize the total number cars passing through. 

### 4. Some RL vulgarization 

* The Q function maps the state action pairs to a cumulative sum of future rewards. 

$Q _ \pi(s,a)$, the cumulative reward function  can be computed via Bellman equation.  

### 5. Reinforcement Learning Model 

#### This section defines what are the states, actions and rewards 

* States: a vehicular network extracts positions and speed of the vehicles. It's basically a grid where we have (position, speed) pairs. Position is a binary value, speed is an integer in m/s.  

* actions: the action space is defined over the durations for the phases. the duration changes of legal phases  between two neighboring cycles is the MDP. 

* rewards: Some definitions first: $i _ t$ denotes the $i $ th observed vehicle. $i $ basically indexes a time varying list $(1 < i _ t < N _ t)$. $N _ t$ denotes the total number vehicles at time t. The reward in the $t $ th cycle is defined by $r _ t = W _ t - W _ t+1$. Where, $W _ t $ is the total wait time of the vehicles in time $t$, that is $ W _ t = \sum _ {i _ t = 1} ^ N _ t w _ {i _ t, t} $, where they use $w _ {i _ t, t} $ to denote the wait time of the $i _ t $ ' th vehicle in cycle $t$. I think they could have just used $w _ {i _ t}$. 


### 6. Double Dueling Deep Q Network 

I am not an expert on Reinforcement Learning. But here are the pre-requisites to get what's happening I think:

* https://arxiv.org/pdf/1803.11115.pdf



## Alexandre Bayen talk on Reinforcement Learning in Transportation: 

https://www.youtube.com/watch?v=gOwm5JMvPQM&feature=youtu.be

This is a very good talk. He shows how traffic efficiency can be ameliorated with Reinforcement Learning. He mentions how the results they are getting supersedes mathematical models that came before. Basically the system is too complex for our little brains to build mathematical models. We instead define reward structures, and let computational exploration find a good solution. Cars circling is a great example for this. 


