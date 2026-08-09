"""xqrobotV2 wheel-legged bipedal robot env package."""

from .joystick import XqRobotWLWalkFlatCfg, XqRobotWLWalkFlatEnv
from .jump import XqRobotWLJumpFlatCfg, XqRobotWLJumpFlatEnv
from .jump_srl import XqRobotWLJumpSRLFlatCfg, XqRobotWLJumpSRLFlatEnv
from .jump_vmc import XqRobotWLJumpVMCFlatCfg, XqRobotWLJumpVMCFlatEnv
from .jump_srl_vmc import XqRobotWLJumpSRLVMCFlatCfg, XqRobotWLJumpSRLVMCFlatEnv
from .rough import XqRobotWLWalkRoughCfg, XqRobotWLWalkRoughEnv
from .stairs import XqRobotWLStairsCfg, XqRobotWLStairsEnv
from .toe_walk import XqRobotWLToeWalkFlatCfg, XqRobotWLToeWalkFlatEnv
