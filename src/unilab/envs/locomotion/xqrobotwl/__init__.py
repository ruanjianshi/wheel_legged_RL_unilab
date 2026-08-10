"""xqrobotV2 wheel-legged bipedal robot env package."""

from .backflip import XqRobotWLBackflipFlatCfg, XqRobotWLBackflipFlatEnv
from .fall_recovery import XqRobotWLFallRecoveryFlatCfg, XqRobotWLFallRecoveryFlatEnv
from .joystick import XqRobotWLWalkFlatCfg, XqRobotWLWalkFlatEnv
from .jump import XqRobotWLJumpFlatCfg, XqRobotWLJumpFlatEnv
from .jump_srl import XqRobotWLJumpSRLFlatCfg, XqRobotWLJumpSRLFlatEnv
from .jump_srl_vmc import XqRobotWLJumpSRLVMCFlatCfg, XqRobotWLJumpSRLVMCFlatEnv
from .jump_vmc import XqRobotWLJumpVMCFlatCfg, XqRobotWLJumpVMCFlatEnv
from .rough import XqRobotWLWalkRoughCfg, XqRobotWLWalkRoughEnv
from .single_leg import XqRobotWLSingleLegFlatCfg, XqRobotWLSingleLegFlatEnv
from .single_leg_move import XqRobotWLSingleLegMoveCfg, XqRobotWLSingleLegMoveEnv
from .single_leg_unicycle import (
    XqRobotWLSingleLegUnicycleCfg,
    XqRobotWLSingleLegUnicycleEnv,
)
from .stairs import XqRobotWLStairsCfg, XqRobotWLStairsEnv
from .toe_walk import XqRobotWLToeWalkFlatCfg, XqRobotWLToeWalkFlatEnv
