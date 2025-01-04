#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly
#show: university-theme.with(aspect-ratio: "16-9")

= 2025.1.3

== LLMs for RL

1. Utilizing LLMs to decompose complex tasks and generate high-level plans

#link("https://github.com/chufanchen/read-paper-and-code/issues/192")[Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks]

#figure(
  grid(
    columns: (auto, auto),
    image("assets/plan-seq-learn.png", width: auto),
    image("assets/plan-seq-learn-algo.png", width: auto),
  ),
  caption: [PSL]
)


#link("https://github.com/chufanchen/read-paper-and-code/issues/204")[KALM: Knowledgeable Agent by Offline Reinforcement Learning from Large Language Model Rollouts]

a. LLM Ground as Instruction-following fine-tuning
- Dynamics prediction
- Rollout explanation
- Rollout generation
- Consequence prediction

b. "Generate a rollout for the following goal: [GOAL]"

c. Train offline RL on both the existing offline dataset and imaginary rollouts

#figure(image("assets/kalm.png", height: auto), caption: [KALM])

2. Using LLMs to design the reward function
#link("https://github.com/chufanchen/read-paper-and-code/issues/203")[Code as Reward: Empowering Reinforcement Learning with VLMs]

a. Prompt the pre-trained VLMs in a sequential manner, incrementally building up the necessary information to specify the reward functions

b. Verification using Expert and Random trajectories

c. Using Generated Programs in the RL loop

#figure(image("assets/vlm-car.png", height: auto), caption: [VLM-CaR])

#link("https://github.com/chufanchen/read-paper-and-code/issues/198")[Learning Reward for Robot Skills Using Large Language Models via Self-Alignment]

a. We extract the skill-specific reward function parameterization from LLM using a sequence of guiding prompts

b. iteratively self-align the reward function $R_theta$ using ranking-based preference learning

#figure(image("assets/rewardselfalign.png", height: auto), caption: [Reward Self-Alignment])

3. Directly training LLMs as the behavior polices with RL algorithms
4. Skill Learning with LLMs
- #link("https://github.com/chufanchen/read-paper-and-code/issues/196")[Language-guided Skill Learning with Temporal Variational Inference]
5. Enhance state representation with LLMs
- #link("https://github.com/chufanchen/read-paper-and-code/issues/197")[LLM-Empowered State Representation for Reinforcement Learning]

6. Guiding Pretraining in Reinforcement Learning with Large Language Models

== Credit Assignment

RL agents where sampling actions from a policy means sampling a sequence of tokens mapping to an action from a (suitably conditioned) large language model. Since actions are typically described as a sequence of more than one token, this introduces issues around *credit assignment* between tokens contributing to an action.

#figure(image("assets/POAD.png", height: 50%), caption: [#text(size: 20pt)[The necessity of aligning language agents with environments to
exclude the wrong option, since the agent does not initially know that “coffee table is empty”; Action-level optimization is uncertain to what extent the key tokens]])

- #link("https://github.com/chufanchen/read-paper-and-code/issues/224")[Reinforcing LLM Agents via Policy Optimization with Action Decomposition]

Bias in value networks (used by PPO) impacts performance.

- #link("https://github.com/chufanchen/read-paper-and-code/issues/216")[VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment]



== Self Correction

#link("https://github.com/chufanchen/read-paper-and-code/issues/222")[Training Language Models to Self-Correct via Reinforcement Learning]

#figure(image("assets/Score.png", height: 50%), caption: [SCoRe: Two-stage RLHF.])

#link("https://github.com/chufanchen/read-paper-and-code/issues/191")[Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization]

a. Actor Model is a frozen LLM to generate actions

b. Retrospective Model: A local LLM to generate reflection feedback

c. Following the RLHF training procedures with PPO $r(x_(k,i),y_(k,i))=G_(k,i+1)-G_(k,i)$. ($x,y$ is a reflection instruction-response pair)

#figure(image("assets/retroformer.png"), caption: [Retroformer.])

== RL for LLMs

=== DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning

- Method: Offline RL + Offline-to-Online RL

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/225")[WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning]

- Method: ORM SFT Training + Curriculum RL

=== Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning

- Motivation: Instruction tunning VLM is not optimal for decision-making tasks.
- Method: CoT and end-to-end RL training

#figure(image("assets/vlm-0.jpg", height: 70%), caption: [Our method provides the VLM with a task description and then asks the model to output a CoT reasoning followed by a text action. Next, the text action will be parsed into an executable action to interact with the environment for task rewards and the next state.])
#figure(image("assets/vlm-1.jpg", width: auto), caption: [Finally, we apply PPO to fine-tune the model with the task rewards.])

=== Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations

- Motivation: Pretrained LLMs can be served as effective "human simulators" to aid in the training of dialogue agents. However, the effectiveness of this approach(e.g. offline RL) is limited by the quality of the dialogue data used for training.
- Method:
  1. A hindsight controller $c_H$ that takes any completed dialogue as input, as well as a prefix of that dialogue, and proposes a different, more preferable action to take
  2. A forward model $hat(P)$ that simulates a hypothetical completed dialogue from any prefix
  3. A reward model $hat(r)$ to assign a reward for any completed dialogue
  4. An offline RL method for learning a policy from a static dataset of dialogues

#figure(image("assets/hindsight.png", height: 70%), caption: [Hindsight Generation.])

=== AGILE: A Novel Reinforcement Learning Framework of LLM Agents

LLM Agents 被定义为一个 token-level MDP（Markov Decision Process）。动作空间（Action space）由 LLM 的词表构成，LLM 生成的每一个 token 是一个动作，LLM 本身则作为 Agent 的策略模型（Policy model）。Agent 的状态（State）由 LLM 上下文和记忆组成。在每个时刻，LLM 预测动作，执行器根据预定义的逻辑完成状态转移，同时环境给予 Agent 相应的奖励（Reward）。

- Method: Imitation learning + Reinforcement learning + Seek for advice

#figure(image("assets/AGILE.png", height: 70%), caption: [AGILE: A Framework unify LLM, memory, tools and executor.])

=== Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

#figure(image("assets/R3.png", height: 70%), caption: [R3: Reverse Curriculum Reinforcement Learning.])

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/217")[Empowering LLM Agents with Zero-Shot Optimal Decision-Making through Q-learning]

- Method: Q-learning + LLM as world model and rollout using MCTS

Many of these problems require the agent to explicitly take the steps to gather information before making a decision. *Single-turn* RL for LLMs cannot learn such nuanced strategies as they attempt to solve the problem within a single step.*Multi-turn* RL for LLMs can become sample inefficient in multi-step settings that require interaction with an external environment.

#figure(image("assets/multiturn.jpeg", height: 70%), caption: [Agents need multi-turn LLM fine-tuning.])

- #link("https://github.com/chufanchen/read-paper-and-code/issues/193")[ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL]


== LLM Reasoning via Planning

#figure(image("assets/RAP.png", height: 50%), caption: [Reasoning with Language Model is Planning with World Model.])

#link("https://github.com/chufanchen/read-paper-and-code/issues/199")[Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models]

// == Speculative Decoding

// #figure(image("assets/spec.png", height: 40%), caption: [Draft-then-Verify.])

// - #link("https://github.com/chufanchen/read-paper-and-code/issues/210")[Block Verification Accelerates Speculative Decoding]
// - #link("https://github.com/chufanchen/read-paper-and-code/issues/206")[SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration]

// == Caching

// #figure(image("assets/diffusion_cache.png", height: 30%), caption: [Cache Method for Diffusion model.])
// - #link("https://github.com/chufanchen/read-paper-and-code/issues/212")[Accelerating Diffusion Transformers with Token-wise Feature Caching]
// - #link("https://github.com/chufanchen/read-paper-and-code/issues/207")[FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality]

// == KV Cache

// - #link("https://github.com/chufanchen/read-paper-and-code/issues/208")[VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration]
// - #link("https://github.com/chufanchen/read-paper-and-code/issues/213")[Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs]

// == Sparse Attention

// - #link("https://github.com/chufanchen/read-paper-and-code/issues/209")[CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation]
// - #link("https://github.com/chufanchen/read-paper-and-code/issues/214")[MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention]