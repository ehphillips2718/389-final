from collections import deque
import socket
import json
import subprocess
import struct
import gymnasium as gym
from os import path
from typing import Callable, Tuple, Dict, Union
from time import sleep, monotonic
from enum import Enum
from gymnasium import spaces
from ..state import FootsiesState, FootsiesBattleState
from ..moves import FootsiesMove, FOOTSIES_MOVE_ID_TO_INDEX
from .exceptions import FootsiesGameClosedError

# TODO: move training agent input reading (through socket comms) to Update() instead of FixedUpdate()
# TODO: dynamically change the game's timeScale value depending on the estimated framerate


class FootsiesEnv(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 60}
    
    STATE_MESSAGE_SIZE_BYTES = 4
    COMM_TIMEOUT = 10

    class RemoteControlCommand(Enum):
        NONE = 0
        RESET = 1
        STATE_SAVE = 2
        STATE_LOAD = 3
        P2_BOT = 4
        SEED = 5

    def __init__(
        self,
        frame_delay: int = 0,
        render_mode: str = None,
        game_path: str = "./Build/FOOTSIES",
        game_address: str = "localhost",
        game_port: int = 11000,
        skip_instancing: bool = False,
        fast_forward: bool = True,
        sync_mode: str = "synced_non_blocking",
        remote_control_port: int = 11002,
        by_example: bool = False,
        opponent: Callable[[dict, dict], Tuple[bool, bool, bool]] = None,
        opponent_port: int = 11001,
        vs_player: bool = False,
        dense_reward: bool = True,
        log_file: str = None,
        log_file_overwrite: bool = False,
    ):
        """
        FOOTSIES training environment

        Parameters
        ----------
        frame_delay: int
            with how many frames of delay should environment states be sent to the agent (meant for human reaction time emulation)
        render_mode: str
            how should the environment be rendered
        game_path: str
            path to the FOOTSIES executable. Preferably a fully qualified path
        game_address: str
            address of the FOOTSIES instance
        game_port: int
            port of the FOOTSIES instance
        skip_instancing: bool
            whether to skip instancing of the game
        fast_forward: bool
            whether to run the game at a much faster rate than normal
        sync_mode: str
            one of "async", "synced_non_blocking" or "synced_blocking":
            - "async": process the game without making sure the agents have provided inputs. Doesn't make much sense to have `fast_forward` enabled as well. Due to non-blocking communications, input may only be received every other frame, slowing down game interaction speed to half
            - "synced_non_blocking": at every time step, the game will wait for all agents' inputs before proceeding. Communications are non-blocking, and as such may have the same problem as above
            - "synced_blocking": similar to above, but communications are blocking. If using human `render_mode`, the game may have frozen rendering. Remote control is not supported in this mode
            
        remote_control_port: int
            the port to which the remote control socket will connect to
        by_example: bool
            whether to simply observe the in-game bot play the game. Actions passed in `step()` are ignored
        opponent: Callable[[dict, dict, bool, bool], Tuple[bool, bool, bool]]
            if not `None`, it's the policy to be followed by the agent's opponent. It's recommended that the environment is `synced` if a policy is supplied, since both the agent and the opponent will be acting at the same time
        opponent_port: int
            if an opponent policy is supplied, then this is the game's port to which the opponent's actions are sent
        vs_player: bool
            whether to play against a human opponent (who will play as P2). It doesn't make much sense to let `fast_forward` be `True`. Not allowed if `opponent` is specified
        dense_reward: bool
            whether to use dense reward on the environment, rather than sparse reward. Sparse reward only rewards the agent on win or loss (1 and -1, respectively). Dense reward rewards the agent on inflicting/receiving guard damage (0.3 and -0.3, respectively), but on win/loss a compensation is given such that the sum is like the sparse reward (1 and -1, respectively)
        log_file: str
            path of the log file to which the FOOTSIES instance logs will be written. If `None` logs will be written to the default Unity location
        log_file_overwrite: bool
            whether to overwrite the specified log file if it already exists

        WARNING: if the environment has an unexpected error or closes incorrectly, it's possible the game process will still be running in the background. It should be closed manually in that case
        """
        valid_sync_modes = {"async", "synced_non_blocking", "synced_blocking"}
        if sync_mode not in valid_sync_modes:
            raise ValueError(
                f"sync mode '{sync_mode}' is invalid, must be one of {valid_sync_modes}"
            )
        if opponent is not None and vs_player:
            raise ValueError(
                "custom opponent and human opponent can't be specified together"
            )

        self.game_path = game_path
        self.game_address = game_address
        self.game_port = game_port
        self.skip_instancing = skip_instancing
        self.fast_forward = fast_forward
        self.sync_mode = sync_mode
        self.remote_control_port = remote_control_port
        self.by_example = by_example
        self.opponent = opponent
        self.opponent_port = opponent_port
        self.vs_player = vs_player
        self.dense_reward = dense_reward
        self.log_file = log_file
        self.log_file_overwrite = log_file_overwrite

        # Create a queue containing the last `frame_delay` frames so that we can send delayed frames to the agent
        # The actual capacity has one extra space to accomodate for the case that `frame_delay` is 0, so that
        # the only state to send (the most recent one) can be effectively sent through the queue
        self.delayed_frame_queue: deque[FootsiesState] = deque(
            [], maxlen=frame_delay + 1
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.comm.settimeout(self.COMM_TIMEOUT)
        self.remote_control_comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.remote_control_comm.settimeout(self.COMM_TIMEOUT)
        self._connected = False
        self._game_instance = None

        self.opponent_comm = (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.opponent is not None
            else None
        )
        if self.opponent_comm is not None:
            self.opponent_comm.settimeout(self.COMM_TIMEOUT)
        self._opponent_connected = False

        # Don't consider the end-of-round moves
        relevant_moves = set(FootsiesMove) - {FootsiesMove.WIN, FootsiesMove.DEAD}
        maximum_move_duration = max(m.value.duration for m in relevant_moves)

        # The observation space is divided into 2 columns, the first for player 1 (the agent) and the second for player 2
        self.observation_space = spaces.Dict(
            {
                "guard": spaces.MultiDiscrete([4, 4]),  # 0..3
                "move": spaces.MultiDiscrete(
                    [len(relevant_moves), len(relevant_moves)]
                ),
                "move_frame": spaces.Box(
                    low=0.0, high=maximum_move_duration, shape=(2,)
                ),
                "position": spaces.Box(low=-4.6, high=4.6, shape=(2,)),
            }
        )

        # 3 actions, which can be combined: left, right, attack
        self.action_space = spaces.MultiBinary(3)

        # -1 for losing, 1 for winning, 0 otherwise
        self.reward_range = (-1, 1)

        # Save the most recent state internally
        # Useful to differentiate between the previous and current environment state
        self._current_state = None

        # The latest observation and info that the agent saw
        # Required in order to communicate to the opponent the same observation and info
        self._most_recent_observation = None
        self._most_recent_info = None

        # Keep track of the total reward during this episode
        # Only used when dense rewards are enabled
        self._cummulative_episode_reward = 0.0

        # Keep track of whether the current episode is finished
        # Necessary when calling reset() when it isn't finished, which will require a hard reset
        self.has_terminated = True

    def _instantiate_game(self):
        """
        Start the FOOTSIES process in the background, with the specified render mode.
        No-op if already instantiated or instantiation is skipped
        """
        if self.skip_instancing:
            return
        
        if self._game_instance is None:
            args = [
                self.game_path,
                "--mute",
                "--training",
                "--p1-address",
                self.game_address,
                "--p1-port",
                str(self.game_port),
                "--remote-control-address",
                self.game_address,
                "--remote-control-port",
                str(self.remote_control_port),
            ]
            if self.render_mode is None:
                args.extend(["-batchmode", "-nographics"])
            if self.fast_forward:
                args.append("--fast-forward")
            
            if self.sync_mode == "synced_non_blocking":
                args.append("--synced-non-blocking")
            elif self.sync_mode == "synced_blocking":
                args.append("--synced-blocking")

            if self.by_example:
                args.append("--p1-bot")
                args.append("--p1-spectator")
            
            if self.vs_player:
                args.append("--p2-player")
            elif self.opponent is None:
                args.append("--p2-bot")
            else:
                args.extend(
                    [
                        "--p2-address",
                        self.game_address,
                        "--p2-port",
                        str(self.opponent_port),
                        "--p2-no-state",
                    ]
                )
            if self.log_file is not None:
                if not self.log_file_overwrite and path.exists(self.log_file):
                    raise FileExistsError(
                        f"the log file '{self.log_file}' already exists and the environment was set to not overwrite it"
                    )
                args.extend(["-logFile", self.log_file])

            self._game_instance = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _socket_connect(self, sckt: socket.socket, address: tuple, retry_delay: float = 0.5):
        connected = False
        while not connected:
            try:
                sckt.connect(address)
                connected = True

            except (ConnectionRefusedError, ConnectionAbortedError):
                sleep(
                    retry_delay
                )  # avoid constantly pestering the game for a connection
                continue

    def _connect_to_game(self, retry_delay: float = 0.5):
        """
        Connect to the FOOTSIES instance specified by the environment's address and port.
        If the connection is refused, wait `retry_delay` seconds before trying again.
        No-op if already connected.

        If an opponent was supplied, then try establishing a connection for the opponent as well.
        """
        if not self._connected:
            self._socket_connect(self.comm, (self.game_address, self.game_port), retry_delay)
            self._socket_connect(self.remote_control_comm, (self.game_address, self.remote_control_port), retry_delay)
            self._connected = True

        if self.opponent is not None:
            if not self._opponent_connected:
                self._socket_connect(self.opponent_comm, (self.game_address, self.opponent_port), retry_delay)
                self._opponent_connected = True

    def _game_recv_bytes(self, sckt: socket.socket, size: int) -> bytes:
        """Receive a message of the given size from the FOOTSIES instance. Raises `FootsiesGameClosedError` if a problem occurred"""
        try:
            res = bytes()
            while len(res) < size:
                res += sckt.recv(size - len(res))
        except TimeoutError:
            raise FootsiesGameClosedError("game took too long to respond, will assume it's closed")

        # The communication is assumed to work correctly, so if a message wasn't received then the game must have closed
        if len(res) == 0:
            raise FootsiesGameClosedError("game has closed")

        return res

    def _game_recv_message(self, sckt: socket.socket) -> str:
        """Receive an UTF-8 message from the given socket"""
        message_size_bytes = self._game_recv_bytes(sckt, self.STATE_MESSAGE_SIZE_BYTES)
        message_size = struct.unpack("!I", message_size_bytes)[0]

        return self._game_recv_bytes(sckt, message_size).decode("utf-8")

    def _receive_and_update_state(self) -> FootsiesState:
        """Receive the environment state from the FOOTSIES instance"""
        state_json = self._game_recv_message(self.comm)

        self._current_state = FootsiesState(**json.loads(state_json))

        return self._current_state

    def _send_action(
        self, action: "tuple[bool, bool, bool]", is_opponent: bool = False
    ):
        """Send an action to the FOOTSIES instance"""
        action_message = bytearray(action)
        try:
            if is_opponent:
                self.opponent_comm.sendall(action_message)
            else:
                self.comm.sendall(action_message)
        except OSError:
            raise FootsiesGameClosedError

    def _extract_obs(self, state: FootsiesState) -> dict:
        """Extract the relevant observation data from the environment state"""
        # Simplify the number of frames since the start of the move for moves that last indefinitely
        p1_move_frame_simple = (
            0
            if state.p1Move
            in {
                FootsiesMove.STAND.value.id,
                FootsiesMove.FORWARD.value.id,
                FootsiesMove.BACKWARD.value.id,
            }
            else state.p1MoveFrame
        )
        p2_move_frame_simple = (
            0
            if state.p2Move
            in {
                FootsiesMove.STAND.value.id,
                FootsiesMove.FORWARD.value.id,
                FootsiesMove.BACKWARD.value.id,
            }
            else state.p2MoveFrame
        )

        return {
            "guard": (state.p1Guard, state.p2Guard),
            "move": (
                FOOTSIES_MOVE_ID_TO_INDEX[state.p1Move],
                FOOTSIES_MOVE_ID_TO_INDEX[state.p2Move],
            ),
            "move_frame": (p1_move_frame_simple, p2_move_frame_simple),
            "position": (state.p1Position, state.p2Position),
        }

    def _extract_info(self, state: FootsiesState, obs: dict) -> dict:
        """Get the current additional info from the environment state"""
        return {
            "frame": state.globalFrame,
            "p1_action": state.p1MostRecentAction,
            "p2_action": state.p2MostRecentAction,
            # Put a copy of the observation in the information dict, so that it's preserved through the wrappers
            **obs
        }

    def _get_sparse_reward(
        self, state: FootsiesState, next_state: FootsiesState, terminated: bool
    ) -> float:
        """Get the sparse reward from this environment step. Equal to 1 or -1 on win/loss, respectively"""
        return (1 if next_state.p2Vital == 0 else -1) if terminated else 0

    def _get_dense_reward(
        self, state: FootsiesState, next_state: FootsiesState, terminated: bool
    ) -> float:
        """Get the dense reward from this environment step. Sums up to 1 or -1 on win/loss, but is also given when inflicting/dealing guard damage (0.3 and -0.3, respectively)"""
        reward = 0.0
        if next_state.p1Guard < state.p1Guard:
            reward -= 0.3
        if next_state.p2Guard < state.p2Guard:
            reward += 0.3

        self._cummulative_episode_reward += reward

        if terminated:
            reward += (
                1 if next_state.p2Vital == 0 else -1
            ) - self._cummulative_episode_reward

        return reward

    def _remote_control_send_command(self, command: RemoteControlCommand, value: str = "") -> "any":
        """
        Send a command to the game.
            
        WARNING: this method is not supported if the environment is in `synced_blocking` mode,
        since the game will be most of the time waiting for the agent to send an action rather
        than waiting for a command.
        """
        if self.sync_mode == "synced_blocking":
            raise RuntimeError("remote control is not supported in 'synced_blocking' mode")
            
        if command == self.RemoteControlCommand.NONE:
            return
        
        message = {"command": command.value, "value": value}
        message_json = json.dumps(message).encode("utf-8")
        
        size_suffix = struct.pack("!I", len(message_json))

        self.remote_control_comm.sendall(size_suffix + message_json)

        if command == self.RemoteControlCommand.STATE_SAVE:
            battle_state_json = self._game_recv_message(self.remote_control_comm)
            return FootsiesBattleState.from_json(battle_state_json)

    def save_battle_state(self) -> FootsiesBattleState:
        """Save the current game state"""
        self._instantiate_game()
        self._connect_to_game()
    
        return self._remote_control_send_command(self.RemoteControlCommand.STATE_SAVE)

    def load_battle_state(self, battle_state: FootsiesBattleState):
        """Make the game load a specific battle state"""
        self._instantiate_game()
        self._connect_to_game()
    
        self._remote_control_send_command(self.RemoteControlCommand.STATE_LOAD, battle_state.json())

    def _request_reset(self):
        """Request an environment reset"""
        self._remote_control_send_command(self.RemoteControlCommand.RESET)

    def _request_opponent_change(self, bot: bool):
        """Request that the game changes player 2 to be either the in-game bot or a remote opponent"""
        self._remote_control_send_command(self.RemoteControlCommand.P2_BOT, str(bot))

    def _request_seed_set(self, seed: int):
        """Request the game to set its random number generator seed to the specified value"""
        self._remote_control_send_command(self.RemoteControlCommand.SEED, str(seed))

    def set_opponent(self, opponent: Callable[[dict], Tuple[bool, bool, bool]]):
        """
        Set the agent's opponent to the specified custom policy, or `None` if the default environment opponent should be used.
        Returns whether the environment requires calling `reset(...)` after calling this method.

        WARNING: the environment needs to be set up with a custom opponent on creation (may be a dummy one), or else this method will raise an exception.
        """
        self._instantiate_game()
        self._connect_to_game()

        # TODO: maybe try making this not a requirement
        if self.opponent_comm is None:
            raise RuntimeError("the environment needs to be created with a custom opponent before calling this method")

        require_request = (opponent is not None and self.opponent is None) or (
            opponent is None and self.opponent is not None
        )

        # Update the internal custom opponent policy
        self.opponent = opponent

        if require_request:
            self._request_opponent_change(bot=self.opponent is None)

    def reset(self, *, seed: int = None, options: dict = None) -> "tuple[dict, dict]":
        super().reset(seed=seed)
        self._instantiate_game()
        self._connect_to_game()

        if seed is not None:
            self._request_seed_set(seed)

        if not self.has_terminated:
            self._request_reset()
        
        self.delayed_frame_queue.clear()
        self._cummulative_episode_reward = 0.0

        first_state = self._receive_and_update_state()
        # Guarantee it's the first environment state
        while first_state.globalFrame != -1:
            first_state = self._receive_and_update_state()
        # We leave a space at the end of the queue since insertion of the most recent state happens before popping the oldest state.
        # This is done so that the case when `frame_delay` is 0 is correctly handled
        while len(self.delayed_frame_queue) < self.delayed_frame_queue.maxlen - 1:
            # Give the agent the same initial state but repeated (`frame_delay` - 1) times
            self.delayed_frame_queue.append(first_state)

        # The episode can't terminate right in the beginning
        # This will also allow reset() to be called right after reset()
        self.has_terminated = False

        obs = self._extract_obs(first_state)
        info = self._extract_info(first_state, obs)
        # Create a copy of this observation (make sure it's not edited because 'obs' was changed afterwards, which may happen with wrappers)
        self._most_recent_observation = obs.copy()
        self._most_recent_info = info.copy()
        return obs, info

    # Step already assumes that the queue of delayed frames is full from reset()
    def step(
        self, action: "tuple[bool, bool, bool]"
    ) -> "tuple[dict, float, bool, bool, dict]":
        # Send action
        if not self.by_example:
            self._send_action(action, is_opponent=False)

        if self.opponent is not None:
            opponent_action = self.opponent(self._most_recent_observation, self._most_recent_info)
            self._send_action(opponent_action, is_opponent=True)

        # Save the state before the environment step for later
        previous_state = self._current_state

        # Store the most recent state first and then take the oldest one
        most_recent_state = self._receive_and_update_state()
        self.delayed_frame_queue.append(most_recent_state)
        state = self.delayed_frame_queue.popleft()

        # In the terminal state, the defeated opponent gets into a move (DEAD) that doesn't occur throughout the game, so in that case we default to STAND
        state.p1Move = (
            state.p1Move
            if state.p1Move
            not in {FootsiesMove.DEAD.value.id, FootsiesMove.WIN.value.id}
            else FootsiesMove.STAND.value.id
        )
        state.p2Move = (
            state.p2Move
            if state.p2Move
            not in {FootsiesMove.DEAD.value.id, FootsiesMove.WIN.value.id}
            else FootsiesMove.STAND.value.id
        )

        # Get next observation, info and reward
        obs = self._extract_obs(state)
        info = self._extract_info(state, obs)

        terminated = most_recent_state.p1Vital == 0 or most_recent_state.p2Vital == 0
        reward = (
            self._get_dense_reward(previous_state, most_recent_state, terminated)
            if self.dense_reward
            else self._get_sparse_reward(previous_state, most_recent_state, terminated)
        )

        # Enable reset() without requesting a forceful reset if episode terminated normally on this step
        self.has_terminated = terminated

        # Create a copy of this observation and info, as is done in reset()
        self._most_recent_observation = obs.copy()
        self._most_recent_info = info.copy()

        # Environment is never truncated
        return obs, reward, terminated, False, info

    def close(self):
        self.comm.close()  # game should close as well after socket is closed
        self.remote_control_comm.close()
        if self.opponent_comm is not None:
            self.opponent_comm.close()
        if self._game_instance is not None:
            self._game_instance.kill()  # just making sure the game is closed

    @property
    def most_recent_observation(self) -> dict:
        """The most recent observation received by the environment after `reset` or `step`."""
        return self._most_recent_observation

    @property
    def most_recent_info(self) -> dict:
        """The most recent info received by the environment after `reset` or `step`."""
        return self._most_recent_info
    
    @staticmethod
    def find_ports(start: int, step: int = 1, stop: Union[int, None] = None) -> Dict[str, int]:
        """Find available ports for a new instance of `FootsiesEnv`. The `psutil` module is required."""
        import psutil
        from itertools import count
    
        closed_ports = {p.laddr.port for p in psutil.net_connections(kind="tcp4")}
        port_iterator = count(start=start, step=step) if stop is None else range(start, stop, step)
        ports = []

        for port in port_iterator:
            if port not in closed_ports:
                ports.append(port)

            if len(ports) >= 3:
                break

        if len(ports) < 3:
            raise RuntimeError(f"could not find 3 free ports for a new FOOTSIES instance (starting at {start} with steps of {step} until {stop})")

        return {
            "game_port": ports[0],
            "opponent_port": ports[1],
            "remote_control_port": ports[2],
        }


if __name__ == "__main__":
    import pprint

    env = FootsiesEnv(
        game_path="Build/FOOTSIES.exe",
        render_mode="human",
        sync_mode="synced_non_blocking",
        vs_player=False,
        fast_forward=False,
        log_file="out.log",
        log_file_overwrite=True,
        frame_delay=0,
        skip_instancing=False,
    )

    # Keep track of how many frames/steps were processed each second so that we can adjust how fast the game runs
    frames = 0
    seconds = 0
    # Multiply the counters by the decay to avoid infinitely increasing counters and prioritize recent values.
    # Set to a value such that the 1000th counter value in the past will have a weight of 1%
    fps_counter_decay = 0.01 ** (1 / 1000)

    episode_counter = 0
    wins_counter = 0

    # Only to test opponent change through the remote control
    use_custom_opponent = False

    try:
        while True:
            terminated, truncated = False, False
            observation, info = env.reset(seed=0)
            while not (terminated or truncated):
                time_current = monotonic()  # for fps tracking
                action = (False, False, False) # env.action_space.sample()
                next_observation, reward, terminated, truncated, info = env.step(action)

                frames = (frames * fps_counter_decay) + 1
                seconds = (seconds * fps_counter_decay) + monotonic() - time_current
                wins_counter += 1 if terminated and reward > 0 else 0
                print(
                    f"Episode {episode_counter:>3} | {0 if seconds == 0 else frames / seconds:>7.2f} fps | P1 {wins_counter / (episode_counter) if episode_counter > 0 else 0:>7.2%} win rate",
                    end="\r",
                )

                ipt = input("What to do? (s: save | l: load | r: reset | o: toggle opponent)\n")
                if ipt == "s":
                    battle_state = env.save_battle_state()
                    pprint.pprint(battle_state, depth=2, indent=1)
                elif ipt == "l":
                    if battle_state is None:
                        print("No battle state has been saved")
                    else:
                        env.load_battle_state(battle_state)
                elif ipt == "r":
                    # Force reset() to be called again
                    truncated = True
                elif ipt == "o":
                    env.set_opponent((lambda d: (False, False, False)) if use_custom_opponent else None)
                    use_custom_opponent = not use_custom_opponent

                action_to_string = lambda t: " ".join(("O" if a else " ") for a in t)
                print(f"P1: {action_to_string(info['p1_action']):} | P2: {action_to_string(info['p2_action'])}")
            episode_counter += 1

    except KeyboardInterrupt:
        print("Training manually interrupted by the keyboard")

    except FootsiesGameClosedError as err:
        print(
            f"Training interrupted due to the game connection being lost: {err}"
        )

    finally:
        env.close()
