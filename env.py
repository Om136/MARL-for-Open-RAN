# env.py



import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

class RANSlicingEnv(gym.Env):
    def _init_(self, num_slices=3, max_bandwidth=100, max_power=100, render_mode=None):
        super(RANSlicingEnv, self)._init_()

        # Number of slices (agents)
        self.num_slices = num_slices

        # State space: [latency, throughput, packet loss] for each slice
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_slices, 3), dtype=np.float32)

        # Action space: [bandwidth, power] allocation for each slice
        self.action_space = spaces.Box(low=0, high=1, shape=(num_slices, 2), dtype=np.float32)

        # Max resource allocation limits
        self.max_bandwidth = max_bandwidth
        self.max_power = max_power

        # Initialize state, steps, and previous allocations
        self.state = np.zeros((num_slices, 3))
        self.total_steps = 0

        # Store allocations for plotting
        self.bandwidth_allocs = []
        self.power_allocs = []
        self.prev_bandwidth_alloc = np.zeros(num_slices)
        self.prev_power_alloc = np.zeros(num_slices)

        # Rendering mode and plot setup
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.save_path = "ran_slicing_plots"
        os.makedirs(self.save_path, exist_ok=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state and counters
        self.state = np.random.rand(self.num_slices, 3)
        self.total_steps = 0

        # Clear previous allocations
        self.bandwidth_allocs = []
        self.power_allocs = []
        self.prev_bandwidth_alloc = np.zeros(self.num_slices)
        self.prev_power_alloc = np.zeros(self.num_slices)

        # Initialize plot for rendering
        if self.render_mode == "human" and self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        return self.state, {}

    def step(self, action):
        # Clip actions to avoid erratic changes
        min_bandwidth = 0.01
        min_power = 0.01
        bandwidth_alloc = np.clip(action[:, 0], min_bandwidth, 1) * self.max_bandwidth
        power_alloc = np.clip(action[:, 1], min_power, 1) * self.max_power

        # Smooth allocations to reduce fluctuations
        bandwidth_alloc = 0.8 * self.prev_bandwidth_alloc + 0.2 * bandwidth_alloc
        power_alloc = 0.8 * self.prev_power_alloc + 0.2 * power_alloc

        # Store current allocations for the next step
        self.bandwidth_allocs.append(bandwidth_alloc)
        self.power_allocs.append(power_alloc)
        self.prev_bandwidth_alloc = bandwidth_alloc
        self.prev_power_alloc = power_alloc

        # SLA Targets
        target_latency = 0.3
        target_throughput = 0.7
        max_packet_loss = 0.05

        # Initialize rewards and penalties
        rewards = np.zeros(self.num_slices)
        penalties = np.zeros(self.num_slices)

        for i in range(self.num_slices):
            latency, throughput, packet_loss = self.state[i]

            # QoS Satisfaction Scores
            latency_score = np.exp(-3 * latency)  # Reward for lower latency
            throughput_score = np.tanh(throughput / target_throughput)  # Reward saturates at 1
            packet_loss_penalty = np.exp(3 * (packet_loss - max_packet_loss))  # Penalty for high packet loss

            # Reward based on QoS and resource efficiency
            rewards[i] = (throughput_score * 15) + (latency_score * 10) - (packet_loss_penalty * 10)

            # Resource usage penalties
            resource_penalty = 0.05 * (bandwidth_alloc[i] / self.max_bandwidth + power_alloc[i] / self.max_power)
            rewards[i] -= resource_penalty

        # Fairness Penalty: Encourage balanced allocation
        fairness_penalty = np.std(bandwidth_alloc) + np.std(power_alloc)
        total_penalty = np.sum(penalties) + fairness_penalty

        # Calculate total reward
        total_reward = np.sum(rewards) - total_penalty

        # Update state (random for now; in practice, it would depend on the environment dynamics)
        self.state = np.random.rand(self.num_slices, 3)
        self.total_steps += 1

        # Termination condition
        terminated = self.total_steps >= 1000
        truncated = False

        return self.state, total_reward, terminated, truncated, {}

    def render(self, mode=None):
        if self.render_mode == "human":
            print(f"\nStep {self.total_steps} - State: {self.state}")

            # Get the latest allocations
            bandwidth_alloc = self.bandwidth_allocs[-1]
            power_alloc = self.power_allocs[-1]
            print(f"Bandwidth Allocation: {bandwidth_alloc}")
            print(f"Power Allocation: {power_alloc}")

            # Update real-time plot
            self.ax.clear()
            bandwidth_allocs = np.array(self.bandwidth_allocs)
            power_allocs = np.array(self.power_allocs)

            for i in range(self.num_slices):
                self.ax.plot(bandwidth_allocs[:, i], label=f"Bandwidth Slice {i}", linestyle='--', marker='o')
                self.ax.plot(power_allocs[:, i], label=f"Power Slice {i}", linestyle='--', marker='x')

            self.ax.set_xlabel("Step")
            self.ax.set_ylabel("Resource Allocations")
            self.ax.legend()
            self.ax.set_title(f"Step {self.total_steps} Resource Allocation")

            # Save plot for each step
            plt.savefig(os.path.join(self.save_path, f"step_{self.total_steps}.png"))
            plt.pause(0.01)

        if self.total_steps >= 1000 and self.render_mode == "human":
            plt.ioff()
            plt.show()