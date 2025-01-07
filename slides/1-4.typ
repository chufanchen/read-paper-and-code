#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/mitex:0.2.4": *
#show: university-theme.with(aspect-ratio: "16-9")

= 2025.1.4

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
=== #link("https://github.com/chufanchen/read-paper-and-code/issues/196")[Language-guided Skill Learning with Temporal Variational Inference]

Methods:  
  1. Given a dataset of expert demonstrations, we query an LLM (only using the goal and actions as input) for an initial segmentation and a language description for each segment.
  2. Temporal variational inference takes in multi-modal data as input to improve upon the segmentation by merging different subsequences into skills.
  3. Online hierarchical RL on new tasks leveraging the learned skills which can greatly shorten the task horizon and help the agent efficiently learn on new tasks

#figure(image("assets/LAST.png"))

5. Enhance state representation with LLMs
=== #link("https://github.com/chufanchen/read-paper-and-code/issues/197")[LLM-Empowered State Representation for Reinforcement Learning]
Method: 
  1. LLM is prompted to generate codes for state representation and intrinsic reward functions.
  2. $K$ state representations and intrinsic rewards ${F_k}^K_(k=1)$, ${G_k}^K_(k=1)$ are sampled from LLM.
  3. During RL training, function $F$ and $G$ are utilized to generate $s^r = F(s)$ for state representations, and $r^i = G(s, s^r)$ for intrinsic rewards.
  4. Finally, Lipschitz constants and episode returns of each candidate serve as feedback metrics for LLM.

#figure(image("assets/LESR.png", height: 70%))

== Credit Assignment

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/224")[Reinforcing LLM Agents via Policy Optimization with Action Decomposition]

RL agents where sampling actions from a policy means sampling a sequence of tokens mapping to an action from a (suitably conditioned) large language model. Since actions are typically described as a sequence of more than one token, this introduces issues around *credit assignment* between tokens contributing to an action.

#figure(image("assets/POAD.png", height: 50%), caption: [#text(size: 20pt)[The necessity of aligning language agents with environments to
exclude the wrong option, since the agent does not initially know that “coffee table is empty”; Action-level optimization is uncertain to what extent the key tokens]])

Method: Token level Bellman Backup

#mitex(`
\begin{aligned}
& Q_\pi\left(o_t, w_t^{1: j-1}, w_t^j\right) \leftarrow\left\{\begin{array}{ll}
0+\gamma_w \max _{w_t^{j+1}} Q_\pi\left(o_t, w_t^{1: j}, w_t^{j+1}\right), & \text { if } j<\left|a_t\right| \\
R\left(o_t, a_t\right)+\gamma_a \max _{w_{t+1}^1} Q_\pi\left(o_{t+1}, w_{t+1}^1\right), & \text { if } j=\left|a_t\right|
\end{array},\right. \\
& V_\pi\left(o_t, w_t^{1: j}\right) \leftarrow \begin{cases}0+\gamma_w V_\pi\left(o_t, w_t^{1: j+1}\right), & \text { if } j<\left|a_t\right| \\
R\left(o_t, a_t\right)+\gamma_a V_\pi\left(o_{t+1}, \emptyset\right), & \text { if } j=\left|a_t\right|\end{cases}
\end{aligned}
`)


=== #link("https://github.com/chufanchen/read-paper-and-code/issues/216")[VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment]

- Motivation: Bias in value networks (used by PPO) impacts performance.

#mitex(`
\hat{V}_{\mathrm{MC}}\left(s_t\right):=\frac{1}{K} \sum_{k=1}^K R\left(\tau^k\right), \quad \text { where } \tau^1, \ldots, \tau^K \sim \pi_\theta\left(\cdot \mid s_t\right) .
`)


== Self Correction

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/222")[Training Language Models to Self-Correct via Reinforcement Learning]

#figure(image("assets/Score.png", height: 50%), caption: [SCoRe: Two-stage RLHF.])

Single Turn:

#mitex(`
\max _\theta \mathbb{E}_{x_t, y_t \sim \pi_\theta\left(\cdot \mid x_t\right)}\left[\hat{r}\left(y_t, y^*\right)-\beta_1 D_{K L}\left(\pi_\theta\left(\cdot \mid x_t\right)| | \pi_{\mathrm{ref}}\left(\cdot \mid x_t\right)\right)\right],
`)

Multi-Turn:

We explicitly fine-tune the base model to produce high-reward responses at the second attempt, while forcing the model to not change its first attempt by constraining it to be close to the base model using a KL-divergence.

#mitex(`
\max _\theta \mathbb{E}_{x_1, y_1 \sim \pi_\theta(\cdot \mid x), y_2 \sim \pi_\theta\left(\cdot \mid\left[x_1, p_1\right]\right)}\left[\widehat{r}\left(y_2, y^*\right)-\beta_2 D_{K L}\left(\pi_\theta\left(\cdot| | x_1\right)| | \pi_{\mathrm{ref}}\left(\cdot \mid x_1\right)\right)\right],
`)
#mitex(`
\max _\theta \mathbb{E}\left[\sum_{i=1}^2 \widehat{r}\left(y_i, y^*\right)-\beta_1 D_{K L}\left(\pi_\theta\left(\cdot \mid x_i\right)| | \pi_{\mathrm{ref}}\left(\cdot \mid \boldsymbol{x}_i\right)\right)\right],
`)


=== Recursive Introspection: Teaching Language Model Agents How to Self-Improve

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/191")[Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization]

a. Actor Model is a frozen LLM to generate actions

b. Retrospective Model: A local LLM to generate reflection feedback

c. Following the RLHF training procedures with PPO $r(x_(k,i),y_(k,i))=G_(k,i+1)-G_(k,i)$. ($x,y$ is a reflection instruction-response pair)

#figure(image("assets/retroformer.png"), caption: [Retroformer.])

== RL for LLMs

=== DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning

// - Method: Offline RL + Offline-to-Online RL
- Motivation: 
  1. It must make use of online interaction data since static demonstration data would not be representative of the task when the model is deployed.
  2. Learning on-the-fly means the approach must learn from multi-turn interaction data from the model itself, a large of chunk of which would consist of failures. Proper mechanisms must be designed to automatically pick out the correct actions while filtering the wrong ones.
- Method: 
  1. Train instruction-level and step-level value function
  2. Filter out trajectories and states using the value function then train the actor on the filtered data

#figure(image("assets/digirl.jpeg"), caption: [DigiRL])

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/225")[WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning]
- Motivation: A typical challenge in training LLM web agents within WebArena is the scarcity of training tasks, resonating with the situation of developing real-world web agents

#figure(image("assets/webrl.png", width: auto))
- Method: 
  1. ORM(Outcome Reward Model) SFT Training: produce binary reward signal for a given task
  2. Self-evolving curriculum RL: Generate new instruction
    1. During the generation step, we use the in-breadth evolving approach to create new instructions. We select instructions the model failed to complete in previous interaction phases as seeds for generating new instructions.
    2. To ensure that the generated instructions are both feasible in the target environment and aligned with the desired difficulty level, we use the critic to evaluate each new instruction by considering its initial state. We select instructions with critic scores between 0.05 and 0.75, ensuring that only tasks meeting our difficulty criteria are retained


=== Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning

- Motivation: Instruction tunning VLM is not optimal for decision-making tasks.
- Method: CoT and end-to-end RL training

#figure(image("assets/vlm-0.jpg", height: 70%), caption: [Our method provides the VLM with a task description and then asks the model to output a CoT reasoning followed by a text action. Next, the text action will be parsed into an executable action to interact with the environment for task rewards and the next state.])
#figure(image("assets/vlm-1.jpg", width: auto), caption: [Finally, we apply PPO to fine-tune the model with the task rewards.])

=== Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations

- Motivation: Pretrained LLMs can be served as effective "human simulators" to aid in the training of dialogue agents. However, the effectiveness of this approach(e.g. offline RL) is limited by the quality of the dialogue data used for training.

#figure(image("assets/hindsight.png", height: 70%), caption: [Hindsight Generation.])

- Method:
  1. A hindsight controller $c_H$ that takes any completed dialogue as input, as well as a prefix of that dialogue, and proposes a different, more preferable action to take
  2. A forward model $hat(P)$ that simulates a hypothetical completed dialogue from any prefix
  3. A reward model $hat(r)$ to assign a reward for any completed dialogue
  4. An offline RL method for learning a policy from a static dataset of dialogues

=== AGILE: A Novel Reinforcement Learning Framework of LLM Agents

- LLM Agents: token-level MDP
- State: LLM context and memory
- Action space: llm vocabulary
- Policy: llm

#figure(image("assets/AGILE.png", height: 70%), caption: [AGILE: A Framework unify LLM, memory, tools and executor.])

Method: Imitation learning -> Reinforcement learning + Seek for advice


=== Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

#figure(image("assets/R3.png", height: 70%), caption: [R3: Reverse Curriculum Reinforcement Learning.])

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/217")[Empowering LLM Agents with Zero-Shot Optimal Decision-Making through Q-learning]

#figure(image("assets/mlaq.png", height: 70%))

- Method: Q-learning + LLM as world model and rollout using MCTS
  1. MLAQ interacts with the environment through the Q-Planner, which is supported by the* domain-specific* memory that extracts a *task-specific* replay buffer for Q-Update
  2. The environment provides a domain description for the agent, where the agent expands memory and replay buffer through LLM-based imaginary interactions

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/193")[ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL]

Many of these problems require the agent to explicitly take the steps to gather information before making a decision. *Single-turn* RL for LLMs cannot learn such nuanced strategies as they attempt to solve the problem within a single step.*Multi-turn* RL for LLMs can become sample inefficient in multi-step settings that require interaction with an external environment.

#figure(image("assets/multiturn.jpeg", height: 70%), caption: [Agents need multi-turn LLM fine-tuning.])

- Motivation: token-level or utterance-level RL is not sufficient for complex tasks that require multi-turn reasoning.
  - Token-level methods face the challenge of an extremely long horizon (number of tokens per round $*$ number of interactions), leading to numerical instabilities and slow convergence
  - Utterance-level methods face the challenge of an exponential action space (exponential in the number of tokens per utterance), resulting in difficulty in optimizing over such large action space

- Method: Hierarchical Multi-Turn RL

  ArCHer for RL with language models can enjoy the best of both worlds, where an off-policy temporal difference learning method can train an *utterance-level value function* at the high level, and any policy gradient algorithm can optimize the token generation at each turn of the interaction at the low level, treating the high-level value function as the terminal reward for that turn. 

#figure(image("assets/archer.jpeg", height: 70%), caption: [Actor-Critic Framework with a Hierarchical Structure.])

=== Training Software Engineering Agents and Verifiers with SWE-Gym

- Baseline:
  - ArCHer
  - Language IQL

=== RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold

使用负样本训练可以提高模型性能并帮助避免虚假关联，并证明它等同于使用每步优势加权强化学习 (RL) 进行训练

=== Reflexion: Language Agents with Verbal Reinforcement Learning

#figure(image("assets/reflexion.png", height: 90%))

== LLM Reasoning via Planning

=== #link("https://github.com/chufanchen/read-paper-and-code/issues/199")[Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models]
#figure(image("assets/RAP.png", height: 50%), caption: [RAP: Reasoning with Language Model is Planning with World Model.])

#figure(image("assets/rap_0.png"), caption: [ToT vs RAP vs LATS])

Method:
- LLM作为代理（Agent）：LLM负责接收环境输入，生成行动，并接收环境反馈。
- LLM作为价值函数（Value Function）：LLM被用来评估不同行动序列的价值，指导搜索算法的选择过程。
- LLM作为优化器（Optimizer）：LLM被用来生成自我反思，帮助优化未来的决策。
- 蒙特卡洛树搜索（MCTS）：MCTS算法用于在可能的行动空间中进行搜索，选择最有价值的行动序列。
- 环境反馈：环境提供的反馈用于评估行动的价值，以及生成自我反思。
- 自我反思：LLM生成的自我反思用于指导未来的搜索，帮助优化决策过程。

== Misc

- Aviral Kumar
- Sergey Levine
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