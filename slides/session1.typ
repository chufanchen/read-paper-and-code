#import "@preview/touying:0.5.3": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

= 2025.1.3

== LLMs for RL

1. Utilizing LLMs to decompose complex tasks and generate high-level plans
- #link("https://github.com/chufanchen/read-paper-and-code/issues/192")[Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/204")[KALM: Knowledgeable Agent by Offline Reinforcement Learning from Large Language Model Rollouts]
2. Using LLMs to design the reward function
- #link("https://github.com/chufanchen/read-paper-and-code/issues/203")[Code as Reward: Empowering Reinforcement Learning with VLMs]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/198")[Learning Reward for Robot Skills Using Large Language Models via Self-Alignment]
3. Directly training LLMs as the behavior polices with RL algorithms
4. Skill Learning with LLMs
- #link("https://github.com/chufanchen/read-paper-and-code/issues/196")[Language-guided Skill Learning with Temporal Variational Inference]
5. Enhance state representation with LLMs
- #link("https://github.com/chufanchen/read-paper-and-code/issues/197")[LLM-Empowered State Representation for Reinforcement Learning]

== Credit Assignment

RL agents where sampling actions from a policy means sampling a sequence of tokens mapping to an action from a (suitably conditioned) large language model. Since actions are typically described as a sequence of more than one token, this introduces issues around *credit assignment* between tokens contributing to an action.

#figure(image("assets/POAD.png", height: 50%), caption: [#text(size: 20pt)[The necessity of aligning language agents with environments to
exclude the wrong option, since the agent does not initially know that “coffee table is empty”; Action-level optimization is uncertain to what extent the key tokens]])

- #link("https://github.com/chufanchen/read-paper-and-code/issues/224")[Reinforcing LLM Agents via Policy Optimization with Action Decomposition]

Bias in value networks (used by PPO) impacts performance.

- #link("https://github.com/chufanchen/read-paper-and-code/issues/216")[VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment]

== Self Correction

#figure(image("assets/Score.png", height: 50%), caption: [SCoRe: Two-stage RLHF.])

- #link("https://github.com/chufanchen/read-paper-and-code/issues/222")[Training Language Models to Self-Correct via Reinforcement Learning]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/191")[Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization]

== RL for LLMs

- Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations

#figure(image("assets/AGILE.png", height: 70%), caption: [AGILE: A Framework unify LLM, memory, tools and executor.])

- AGILE: A Novel Reinforcement Learning Framework of LLM Agents

#figure(image("assets/R3.png", height: 70%), caption: [R3: Reverse Curriculum Reinforcement Learning.])

- Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

Many of these problems require the agent to explicitly take the steps to gather information before making a decision. *Single-turn* RL for LLMs cannot learn such nuanced strategies as they attempt to solve the problem within a single step.*Multi-turn* RL for LLMs can become sample inefficient in multi-step settings that require interaction with an external environment.

#figure(image("assets/multiturn.jpeg", height: 70%), caption: [Agents need multi-turn LLM fine-tuning.])

- #link("https://github.com/chufanchen/read-paper-and-code/issues/193")[ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL]


== LLM Reasoning via Planning

#figure(image("assets/RAP.png", height: 50%), caption: [An LLM-based policy generates actions and an LLM-based World Model predicts next state under state-action pairs.])

- Reasoning with Language Model is Planning with World Model
- #link("https://github.com/chufanchen/read-paper-and-code/issues/217")[Empowering LLM Agents with Zero-Shot Optimal Decision-Making through Q-learning]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/199")[Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models]

== Speculative Decoding

#figure(image("assets/spec.png", height: 40%), caption: [Draft-then-Verify.])


- #link("https://github.com/chufanchen/read-paper-and-code/issues/210")[Block Verification Accelerates Speculative Decoding]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/206")[SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration]

== Caching

#figure(image("assets/diffusion_cache.png", height: 30%), caption: [Cache Method for Diffusion model.])
- #link("https://github.com/chufanchen/read-paper-and-code/issues/212")[Accelerating Diffusion Transformers with Token-wise Feature Caching]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/207")[FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality]

== KV Cache

- #link("https://github.com/chufanchen/read-paper-and-code/issues/208")[VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/213")[Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs]

== Sparse Attention

- #link("https://github.com/chufanchen/read-paper-and-code/issues/209")[CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation]
- #link("https://github.com/chufanchen/read-paper-and-code/issues/214")[MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention]