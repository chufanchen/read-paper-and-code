#import "@preview/touying:0.5.3": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

= 2025.1.3

== LLMs for RL

1. Utilizing LLMs to decompose complex tasks and generate high-level plans
2. Using LLMs to design the reward function
3. Directly training LLMs as the behavior polices with RL algorithms

- #link("https://github.com/chufanchen/read-paper-and-code/issues/192")[Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/198")[Learning Reward for Robot Skills Using Large Language Models via Self-Alignment]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/196")[Language-guided Skill Learning with Temporal Variational Inference]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/197")[LLM-Empowered State Representation for Reinforcement Learning]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/203")[Code as Reward: Empowering Reinforcement Learning with VLMs]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/204")[KALM: Knowledgeable Agent by Offline Reinforcement Learning from Large Language Model Rollouts]

#figure(image("assets/kalm.png"), caption: [KALM mainly consists of three steps: (1) LLM grounding, (2) Rollout generation, and (3) offline RL.])

== Credit Assignment

Large Language Models (LLMs) are increasingly being applied to complex reasoning tasks, ranging from solving math problems to developing code. The most common method for training these models is Proximal Policy Optimization (PPO), which addresses the *credit assignment* problem using a value network. However, this value network can be significantly biased, which may impact performance.

- #link("https://github.com/chufanchen/read-paper-and-code/issues/216")[VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment]

Misalignment issues hinder LLM-based agent's ability to complete decision-making tasks. Recent advances(GALM and TWOSOME) have showcased that misalignment can be alleviated through building the bridge between optimizing action and optimizing tokens: RL agents where sampling actions from a policy means sampling a sequence of tokens mapping to an action from a (suitably conditioned) large language model. Since actions are typically described as a sequence of more than one token, this introduces issues around *credit assignment* between tokens contributing to an action.

- #link("https://github.com/chufanchen/read-paper-and-code/issues/224")[Reinforcing LLM Agents via Policy Optimization with Action Decomposition]

== Self Correction

- #link("https://github.com/chufanchen/read-paper-and-code/issues/222")[Training Language Models to Self-Correct via Reinforcement Learning]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/191")[Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization]

== Multi-turn RL for LLMs

Many of these problems require the agent to explicitly take the steps to gather information before making a decision. Single-turn RL for LLMs cannot learn such nuanced trategies as they attempt to solve the problem within a single step.

Multi-turn RL for LLMs can become sample inefficient in multi-step settings that require interaction with an external environment. Existing methods either consider a single token as an action or consider an utterance as a single action.

- #link("https://github.com/chufanchen/read-paper-and-code/issues/193")[ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL]

== Safety

- #link("https://github.com/chufanchen/read-paper-and-code/issues/219")[RL, but don't do anything I wouldn't do]

== LLM Reasoning via Planning

#figure(image("assets/RAP.png"), caption: [An LLM-based policy generates actions and an LLM-based World Model predicts next state under state-action pairs.])

- Reasoning with Language Model is Planning with World Model
- #link("https://github.com/chufanchen/read-paper-and-code/issues/217")[Empowering LLM Agents with Zero-Shot Optimal Decision-Making through Q-learning]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/199")[Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models]
