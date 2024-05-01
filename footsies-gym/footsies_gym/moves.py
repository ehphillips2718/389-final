from enum import Enum
from dataclasses import dataclass

@dataclass
class FootsiesMoveInfo:
    id:         int
    duration:   int
    startup:    int
    active:     int
    recovery:   int

class FootsiesMove(Enum):
    STAND = FootsiesMoveInfo(0, 24, 0, 0, 0)
    FORWARD = FootsiesMoveInfo(1, 24, 0, 0, 0)
    BACKWARD = FootsiesMoveInfo(2, 24, 0, 0, 0)
    DASH_FORWARD = FootsiesMoveInfo(10, 16, 0, 0, 0)
    DASH_BACKWARD = FootsiesMoveInfo(11, 22, 0, 0, 0)
    N_ATTACK = FootsiesMoveInfo(100, 22, 4, 2, 16)
    B_ATTACK = FootsiesMoveInfo(105, 21, 3, 3, 15)
    N_SPECIAL = FootsiesMoveInfo(110, 44, 11, 4, 29)
    B_SPECIAL = FootsiesMoveInfo(115, 55, 2, 6, 47)
    DAMAGE = FootsiesMoveInfo(200, 17, 0, 0, 0)
    GUARD_M = FootsiesMoveInfo(301, 23, 0, 0, 0)
    GUARD_STAND = FootsiesMoveInfo(305, 15, 0, 0, 0)
    GUARD_CROUCH = FootsiesMoveInfo(306, 15, 0, 0, 0)
    GUARD_BREAK = FootsiesMoveInfo(310, 36, 0, 0, 0)
    GUARD_PROXIMITY = FootsiesMoveInfo(350, 1, 0, 0, 0)
    DEAD = FootsiesMoveInfo(500, 500, 0, 0, 0)
    WIN = FootsiesMoveInfo(510, 33, 0, 0, 0)

    def in_recovery(self, frame: int) -> bool:
        return frame >= (self.value.startup + self.value.active)
    
    def in_active(self, frame: int) -> bool:
        return self.value.startup <= frame < (self.value.startup + self.value.active)

    def in_startup(self, frame: int) -> bool:
        return frame < self.value.startup
    
    def in_state(self, frame: int) -> int:
        if self.value.startup == 0:
            return 0
        elif self.in_startup(frame):
            return 1
        elif self.in_active(frame):
            return 2
        else:
            return 3

# Helper structures to simplify move IDs (0, 1, 2, ...)
FOOTSIES_MOVE_INDEX_TO_MOVE = list(FootsiesMove)
FOOTSIES_MOVE_ID_TO_INDEX = {move.value.id: i for i, move in enumerate(FOOTSIES_MOVE_INDEX_TO_MOVE)}
