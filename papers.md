# The papers we read 

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

	* the duration for which one uses a status is called a phase. the (possible?) number of phases is the number of legal statuses. 

	* phases cyclically change in a fixed sequence to guide vehicles to pass the intersection. if we have a circular phase structure, we call that a cycle. 

	* the sequence of phases in a cycle is fixed, but the duration of each phase is adaptive.  

* The general idea is to maximize the total number cars passing through. 


