import json
import dataclasses
from typing import List


@dataclasses.dataclass
class FootsiesState:
    """The environment state of FOOTSIES, obtained directly from the game. Less general than `FootsiesBattleState`"""

    p1Vital: int
    p2Vital: int
    p1Guard: int
    p2Guard: int
    p1Move: int
    p2Move: int
    p1MoveFrame: int
    p2MoveFrame: int
    p1Position: float
    p2Position: float
    globalFrame: int
    p1MostRecentAction: "tuple[bool, bool, bool]"
    p2MostRecentAction: "tuple[bool, bool, bool]"

    def __post_init__(self):
        self.p1MostRecentAction = (
            (self.p1MostRecentAction & 1) != 0,
            (self.p1MostRecentAction & 2) != 0,
            (self.p1MostRecentAction & 4) != 0,
        )
        self.p2MostRecentAction = (
            (self.p2MostRecentAction & 1) != 0,
            (self.p2MostRecentAction & 2) != 0,
            (self.p2MostRecentAction & 4) != 0,
        )

    @staticmethod
    def from_battle_state(battle_state: "FootsiesBattleState") -> "FootsiesState":
        return FootsiesState(
            p1Vital=battle_state.p1State.vitalHealth,
            p2Vital=battle_state.p2State.vitalHealth,
            p1Guard=battle_state.p1State.guardHealth,
            p2Guard=battle_state.p2State.guardHealth,
            p1Move=battle_state.p1State.currentActionID,
            p2Move=battle_state.p2State.currentActionID,
            p1MoveFrame=battle_state.p1State.currentActionFrame,
            p2MoveFrame=battle_state.p2State.currentActionFrame,
            p1Position=battle_state.p1State.position[0],
            p2Position=battle_state.p2State.position[0],
            globalFrame=battle_state.frameCount,
            p1MostRecentAction=battle_state.p1State.input[0],
            p2MostRecentAction=battle_state.p2State.input[0],
        )

    def __str__(self):
        """Detailed representation of the environment state"""
        return f"""[P1]:
- Vital: {self.p1Vital}
- Guard: {self.p1Guard}
- Move: {self.p1Move}
- Move frame: {self.p1MoveFrame}
- Position: {self.p1Position}
[P2]:
- Vital: {self.p2Vital}
- Guard: {self.p2Guard}
- Move: {self.p2Move}
- Move frame: {self.p2MoveFrame}
- Position: {self.p2Position}
[Info]:
- Frame: {self.globalFrame}
- P1 most recent action: {self.p1MostRecentAction}
- P2 most recent action: {self.p2MostRecentAction}"""
    

@dataclasses.dataclass
class FootsiesBattleState:
    """The full state of FOOTSIES at a particular time step, meant for saving/loading game states"""
    
    p1State: "FootsiesFighterState"
    p2State: "FootsiesFighterState"
    
    roundStartTime: float
    frameCount: int

    @staticmethod
    def from_json(battle_state_json: str) -> "FootsiesBattleState":
        battle_state_dict = json.loads(battle_state_json)

        return FootsiesBattleState(
            p1State=FootsiesFighterState(**battle_state_dict["p1State"]),
            p2State=FootsiesFighterState(**battle_state_dict["p2State"]),
            roundStartTime=battle_state_dict["roundStartTime"],
            frameCount=battle_state_dict["frameCount"],
        )

    def json(self) -> str:        
        return json.dumps(dataclasses.asdict(self))


@dataclasses.dataclass
class FootsiesFighterState:
    """The full state of one player of FOOTSIES at a particular time step, meant for saving/loading game states"""

    position:       List[int]
    velocity_x:     float
    isFaceRight:    bool

    hitboxes:   List[dict]
    hurtboxes:  List[dict]
    pushbox:    List[dict]

    vitalHealth:    int
    guardHealth:    int

    currentActionID:        int
    currentActionFrame:     int
    currentActionHitCount:  int

    currentHitStunFrame:    int

    input:      List[int]
    inputDown:  List[int]
    inputUp:    List[int]

    isInputBackward:            bool
    isReserveProximityGuard:    bool

    bufferActionID:         int
    reserveDamageActionID:  int

    spriteShakePosition:    int
    maxSpriteShakeFrame:    int

    hasWon:     bool
