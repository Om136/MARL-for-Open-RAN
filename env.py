import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

class RANSlicingEnv(gym.Env):
    def __init__(self, num_slices=3, max_bandwidth=100, max_power=100, render_mode=None):
        super(RANSlicingEnv, self).__init__()

        # Number of slices (agents)
        self.num_slices = num_slices

        # State space: [latency, throughput, packet loss] for each slice
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_slices, 3), dtype=np.float32)

        # Action space: [bandwidth, power] allocation for each slice
        self.action_space = spaces.Box(low=0, high=1, shape=(num_slices, 2), dtype=np.float32)

        # Max resource allocation limits
        self.max_bandwidth = max_bandwidth
        self.max_power = max_power

        # Initialize the state of the environment
        self.state = np.zeros((num_slices, 3))
        self.total_steps = 0  # Track total steps for dynamic adjustments

        # Lists to store bandwidth and power allocation values
        self.bandwidth_allocs = []
        self.power_allocs = []

        # Rendering mode
        self.render_mode = render_mode
        self.fig, self.ax = None, None  # For plotting

        # Folder for saving the plots
        self.save_path = "ran_slicing_plots"
        os.makedirs(self.save_path, exist_ok=True)  # Create folder if it doesn't exist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset network conditions (latency, throughput, packet loss)
        self.state = np.random.rand(self.num_slices, 3)
        self.total_steps = 0  # Reset step count

        # Clear stored allocations
        self.bandwidth_allocs = []
        self.power_allocs = []

        # Initialize plot for real-time rendering
        if self.render_mode == "human" and self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()  # Enable interactive mode for real-time plotting

        return self.state, {}

    def step(self, action):
        # Action: [bandwidth, power] for each slice
        bandwidth_alloc = action[:, 0] * self.max_bandwidth
        power_alloc = action[:, 1] * self.max_power

        # Store allocations for plotting
        self.bandwidth_allocs.append(bandwidth_alloc)
        self.power_allocs.append(power_alloc)

        reward = 0
        penalty = 0

        # Reward calculation based on SLA satisfaction and penalties
        for i in range(self.num_slices):
            latency = self.state[i, 0]
            throughput = self.state[i, 1]
            packet_loss = self.state[i, 2]

            # Reward for maximizing throughput, minimizing latency and packet loss
            slice_reward = (throughput * 10) - (latency * 5) - (packet_loss * 5)
            reward += slice_reward

            # Add penalties for exceeding bandwidth or power limits
            if bandwidth_alloc[i] > self.max_bandwidth or power_alloc[i] > self.max_power:
                penalty += 10  # Penalize for exceeding resource limits

        reward -= penalty  # Subtract penalties from reward

        # Update network conditions (dynamically simulating changing network loads)
        self.state = np.random.rand(self.num_slices, 3)
        self.total_steps += 1

        # Done condition: for example, if the environment runs for 1000 steps
        terminated = self.total_steps >= 1000
        truncated = False  # You can define conditions for truncation if needed

        return self.state, reward, terminated, truncated, {}

    def render(self, mode=None):
        if self.render_mode == "human":
            # Print the state for each step
            print(f"\nStep {self.total_steps} - State: {self.state}")
            
            # Print bandwidth and power allocations
            bandwidth_alloc = self.bandwidth_allocs[-1]  # Get the most recent bandwidth allocation
            power_alloc = self.power_allocs[-1]  # Get the most recent power allocation
            print(f"Bandwidth Allocation: {bandwidth_alloc}")
            print(f"Power Allocation: {power_alloc}")

            # Update real-time plot for latency, throughput, and packet loss
            self.ax.clear()  # Clear the previous plot
            # self.ax.plot(self.state[:, 0], label="Latency", marker='o')
            # self.ax.plot(self.state[:, 1], label="Throughput", marker='x')
            # self.ax.plot(self.state[:, 2], label="Packet Loss", marker='s')

            # Add bandwidth and power plots
            bandwidth_allocs = np.array(self.bandwidth_allocs)
            power_allocs = np.array(self.power_allocs)
            for i in range(self.num_slices):
                self.ax.plot(bandwidth_allocs[:, i], label=f"Bandwidth Slice {i}", linestyle='--', marker='o')
                self.ax.plot(power_allocs[:, i], label=f"Power Slice {i}", linestyle='--', marker='x')

            self.ax.set_xlabel("Slice")
            self.ax.set_ylabel("QoS Metrics and Allocations")
            self.ax.legend()
            self.ax.set_title(f"Step {self.total_steps} QoS Metrics and Resource Allocation Visualization")
            
            # Save the plot at each step
            plt.savefig(os.path.join(self.save_path, f"step_{self.total_steps}.png"))

            plt.pause(0.01)  # Pause for real-time updating

        # Optionally show the final plot at the end of the simulation
        if self.total_steps >= 1000 and self.render_mode == "human":
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Show the final plot
