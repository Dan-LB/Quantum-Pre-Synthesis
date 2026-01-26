import os
import time
import json
import yaml
from copy import deepcopy
from qiskit.circuit.random import random_circuit
from scripts.utils import compute_fidelity
from scripts.agents.RL_env import QuantumMergeEnv
from scripts.mergers import apply_sequence_merge, remove_barriers, remove_identities
from scripts.synthesis.global_synthesis import clifford_t_synthesis
from scripts.agents.utils import get_possible_merges
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from itertools import combinations

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from scripts.agents.base_agent import BaseOptimizer

import csv

class RLMergeOptimizer(BaseOptimizer):
    def __init__(self,  synthesis_config, model_config,
                    task_config=None,
                target_circuit_and_name=(None, None),

                 KAK_SYNTHESIS_CACHE=None, U3_SYNTHESIS_CACHE=None,
                 warmstart=False, greedyplan=None,
                 load_from_WS=False):
        """
        RL-based merge optimizer.

        Args:
            task_config, synthesis_config, model_config: same as GreedyMergeOptimizer.
        """


        super().__init__(synthesis_config,
                            model_config,
                            task_config,
                            target_circuit_and_name,
                            KAK_SYNTHESIS_CACHE,
                            U3_SYNTHESIS_CACHE)


        self.warmstart = warmstart
        self.greedyplan = greedyplan
        self.load_from_WS = load_from_WS
        if self.warmstart:
            assert self.greedyplan is not None, "greedyplan must be provided for warmstart"
            print(f"Warmstart enabled with greedy plan: {self.greedyplan}")


    def optimizer_name(self):
        return "RL"
    


    def optimize(self, verbose=False):
        start_time = time.time()
        if self.warmstart:
            assert False, "Warmstart not supported currently"
            print("Using warmstart environment...")
            env = DummyVecEnv([lambda: QuantumMergeWarmStartEnv(deepcopy(self.base_circuit), self.epsilon,
                                                                greedyplan=self.greedyplan,
                                                                warmstart=True,)])
        else:
            env = DummyVecEnv([lambda: QuantumMergeEnv(deepcopy(self.base_circuit), 
                                                    Synthesizer=self.synthesizer)])
            eval_env = DummyVecEnv([lambda: QuantumMergeEnv(deepcopy(self.base_circuit), 
                                                            Synthesizer=self.synthesizer)])


        # Config values
        INITIAL_STEPS = self.model_config.get("INITIAL_STEPS", 248)
        BATCH_SIZE = self.model_config.get("BATCH_SIZE", 128)
        TOTAL_TIMESTEPS = self.model_config.get("TOTAL_TIMESTEPS", 10_000)
        ENTROPY_COEFFICIENT = self.model_config.get("ENTROPY_COEFFICIENT", 0.05)
        AGENT_CODE = self.model_config.get("AGENT_CODE", "PPO")
        EVAL_BEST_OF = self.model_config.get("EVAL_BEST_OF", 10)

        # Define model path
        model_path = os.path.join(self.exp_path, "model.zip")

        if not self.load_from_WS:
            # Load or create model
            if os.path.exists(model_path):
                if verbose:
                    print("Loading existing model...")
                model = PPO.load(model_path, env=env, device="cpu")
                if verbose:
                    print("Updating path...")
                self.exp_path = self.exp_path.replace("RL", "RL_extended")
                os.makedirs(os.path.join(self.exp_path, "configs"), exist_ok=True)
            else:
                if verbose:
                    print("Creating new model...")
                if AGENT_CODE == "PPO" or AGENT_CODE == "WarmStart":
                    model = PPO("MlpPolicy",
                                env,
                                verbose=0,
                                n_steps=INITIAL_STEPS,
                                batch_size=BATCH_SIZE,
                                ent_coef=ENTROPY_COEFFICIENT,
                                device="cpu")
                else:
                    raise ValueError(f"Unsupported agent code: {AGENT_CODE}")
        else:
            if verbose:
                print("Loading model from warmstart...")
            #replace WS+PPO with WarmStart in path
            WarmStart_path = model_path.replace("WS+PPO", "WarmStart")
            if os.path.exists(WarmStart_path):
                model = PPO.load(WarmStart_path, env=env, device="cpu")
            else:
                raise FileNotFoundError(f"WarmStart model not found at {WarmStart_path}")

        max_no_imp = 10 #10
        # Set up callbacks
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=max_no_imp,
            min_evals=5,#5
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=10_000,#10_000, 
            callback_after_eval=stop_train_callback,
            verbose=1
        )

        # Train (or continue training)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=eval_callback)
        if verbose:
            print("Training completed.")

        # Save updated model
        model.save(model_path)

        # Extract episode logs from environment
        episode_logs = env.envs[0].info
        if verbose:
            print(episode_logs)
            print(f"Number of episodes in this run: {len(episode_logs)}")

        # Prepare path for episode logs
        log_path = os.path.join(self.exp_path, "episode_logs.csv")
        file_exists = os.path.exists(log_path)

        # Append to logs
        with open(log_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["r", "l"])
            if not file_exists:
                writer.writeheader()
            for row in episode_logs:
                writer.writerow(row)

        # Evaluation loop to find best result
        best_t = float('inf')
        best_seq = []
        all_results = []

        for trial in range(EVAL_BEST_OF):
            obs = env.reset()
            done = False
            sequence = []
            step_id = 0

            while not done:
                action, _ = model.predict(obs)
                i, j = env.envs[0].decode_action(action[0])

                if (i, j) in env.envs[0].applied_merges:
                    done = True
                else:
                    sequence.append((i, j))
                    step_start = time.time()
                    obs, reward, done, _ = env.step(action)
                    step_end = time.time()

                    self._log_step(step_id, env.envs[0].t_count,
                                step_end - step_start,
                                step_end - start_time)
                    step_id += 1

            final_t = env.envs[0].t_count
            all_results.append((sequence, final_t))

            if final_t < best_t:
                best_t = final_t
                best_seq = sequence

            print(f"Run {trial + 1}/{EVAL_BEST_OF} — Final T-count: {final_t}, Sequence length: {len(sequence)}")

        self.best_tcount = best_t
        self.applied_merges = best_seq
        self.duration_sec = time.time() - start_time
        self.duration_min = self.duration_sec / 60
        self.optimized_circuit = apply_sequence_merge(deepcopy(self.base_circuit), self.applied_merges)
        self.conclude_optimization()
        best_sequence, best_seq_tcount = self.applied_merges, self.best_tcount


        return best_sequence, best_seq_tcount


    def get_optimized_circuit(self):
        return self.optimized_circuit
