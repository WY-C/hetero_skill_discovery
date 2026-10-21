from .odis_agent import ODISAgent
from .hissd_agent import HISSDAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_hissd"] = HISSDAgent

from .hissd_capability_team_agent import HISSDCapabilityTeamAgent
REGISTRY["mt_hissd_capability_team"] = HISSDCapabilityTeamAgent
