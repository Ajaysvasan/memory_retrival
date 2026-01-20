import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from core.contracts.decision import Decision, ActionType
from logger import Logger
from config import Config


class RLNetwork(nn.Module):
    """Neural network for RL agent"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        return self.network(state)


class RLAgent:
    """Reinforcement Learning agent for system control"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.logger = Logger().get_logger("RLAgent")

        # State and action dimensions
        self.state_dim = 10  # [cache_hit_rate, avg_latency, memory_usage, ...]
        self.action_dim = 4  # [USE_CACHE, RETRIEVE_ANN, EXPAND_TOPIC_SET, REFUSE]

        # Device setup with hardware safety
        self.device = self._setup_device()

        # Initialize network
        self.network = RLNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = RLNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=Config.RL_LEARNING_RATE
        )

        # RL parameters
        self.gamma = Config.RL_GAMMA
        self.epsilon = Config.RL_EPSILON

        # Experience buffer
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = 10000

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_count = 0

        # Load saved model if exists
        self._load_model()

        self.logger.info(f"RL Agent initialized on {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup compute device with hardware safety"""
        if Config.FORCE_CPU:
            self.logger.info("CPU mode forced by configuration")
            return torch.device("cpu")

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

        # Test CUDA usability
        try:
            test_tensor = torch.zeros(Config.GPU_TEST_TENSOR_SIZE).cuda()
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()

            self.logger.info("GPU test passed, using CUDA")
            return torch.device("cuda")

        except Exception as e:
            self.logger.warning(f"GPU test failed: {e}, falling back to CPU")
            return torch.device("cpu")

    def select_action(
        self, state_features: np.ndarray, epsilon_greedy: bool = True
    ) -> Decision:
        """Select action using epsilon-greedy policy"""

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)

        # Epsilon-greedy exploration
        if epsilon_greedy and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
            confidence = self.epsilon
        else:
            with torch.no_grad():
                q_values = self.network(state_tensor)
                action_idx = q_values.argmax(dim=1).item()

                # Softmax for confidence
                probs = torch.softmax(q_values, dim=1)
                confidence = probs[0, action_idx].item()

        # Map index to action
        action_mapping = {
            0: ActionType.USE_CACHE,
            1: ActionType.RETRIEVE_ANN,
            2: ActionType.EXPAND_TOPIC_SET,
            3: ActionType.REFUSE,
        }

        action = action_mapping[action_idx]

        decision = Decision(
            action=action,
            confidence=float(confidence),
            state_features=tuple(state_features.tolist()),
        )

        self.logger.debug(
            f"Selected action: {action.value} (confidence={confidence:.3f})"
        )
        return decision

    def make_decisions(self, state_features: np.ndarray) -> Dict[str, bool]:
        """Make multiple system decisions"""

        # Get Q-values for all actions
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.network(state_tensor).cpu().numpy()[0]

        # Convert Q-values to binary decisions
        # Use threshold for each action independently
        threshold = 0.0  # Positive Q-value means "do it"

        decisions = {
            "use_cache": q_values[0] > threshold,
            "use_ann": q_values[1] > threshold,
            "expand_topics": q_values[2] > threshold,
            "refuse": q_values[3] > threshold,
        }

        return decisions

    def compute_reward(
        self, correct: bool, latency_ms: float, memory_mb: float, hallucinated: bool
    ) -> float:
        """Compute reward signal"""

        # Base reward components
        r_correct = Config.RL_ALPHA_CORRECT if correct else -Config.RL_ALPHA_CORRECT
        r_latency = -Config.RL_BETA_LATENCY * (
            latency_ms / 1000.0
        )  # Normalize to seconds
        r_memory = -Config.RL_GAMMA_MEMORY * (memory_mb / 1024.0)  # Normalize to GB
        r_hallucination = -Config.RL_DELTA_HALLUCINATION if hallucinated else 0.0

        # Total reward
        reward = r_correct + r_latency + r_memory + r_hallucination

        # Clip reward
        reward = np.clip(reward, -Config.RL_REWARD_CLIP, Config.RL_REWARD_CLIP)

        self.logger.debug(
            f"Reward: {reward:.3f} (correct={correct}, latency={latency_ms:.1f}ms)"
        )
        return reward

    def store_experience(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer"""

        experience = (state, action_idx, reward, next_state, done)
        self.replay_buffer.append(experience)

        # Limit buffer size
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step"""

        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        # Unpack batch
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)

        # Compute current Q-values
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.network.state_dict())
        self.logger.debug("Updated target network")

    def save_model(self):
        """Save model to disk"""
        try:
            checkpoint = {
                "network_state": self.network.state_dict(),
                "target_network_state": self.target_network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
                "episode_rewards": self.episode_rewards[-100:],  # Last 100 episodes
            }

            torch.save(checkpoint, self.model_path)
            self.logger.info(f"Saved RL model to {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save RL model: {e}")

    def _load_model(self):
        """Load model from disk"""
        if not self.model_path.exists():
            self.logger.info("No saved RL model found, using fresh initialization")
            return

        try:
            # Load to CPU first for safety
            checkpoint = torch.load(self.model_path, map_location="cpu")

            # Load network states
            self.network.load_state_dict(checkpoint["network_state"])
            self.target_network.load_state_dict(checkpoint["target_network_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Load metadata
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.episode_count = checkpoint.get("episode_count", 0)
            self.episode_rewards = checkpoint.get("episode_rewards", [])

            # Move to device
            self.network.to(self.device)
            self.target_network.to(self.device)

            self.logger.info(
                f"Loaded RL model from {self.model_path} (episodes={self.episode_count})"
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to load RL model: {e}, using fresh initialization"
            )

    def extract_state_features(self, context: Dict) -> np.ndarray:
        """Extract state features from context"""

        features = np.zeros(self.state_dim)

        # Feature 0: Cache hit rate
        features[0] = context.get("cache_hit_rate", 0.0)

        # Feature 1: Average latency (normalized)
        features[1] = min(1.0, context.get("avg_latency_ms", 0.0) / 1000.0)

        # Feature 2: Memory usage (normalized)
        features[2] = min(1.0, context.get("memory_mb", 0.0) / 1024.0)

        # Feature 3: Intent confidence
        features[3] = context.get("intent_confidence", 0.5)

        # Feature 4: Query complexity (normalized word count)
        features[4] = min(1.0, context.get("query_length", 0) / 100.0)

        # Feature 5: Number of available chunks (normalized)
        features[5] = min(1.0, context.get("available_chunks", 0) / 1000.0)

        # Feature 6: Evidence score from last query
        features[6] = context.get("last_evidence_score", 0.0)

        # Feature 7: Contradiction rate
        features[7] = context.get("contradiction_rate", 0.0)

        # Feature 8: Topic diversity
        features[8] = context.get("topic_diversity", 0.5)

        # Feature 9: Time since last query (normalized)
        features[9] = min(1.0, context.get("time_since_last_query_s", 0.0) / 3600.0)

        return features
