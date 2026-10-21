from .odis_learner import ODISLearner
from .hissd_learner import HISSDLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["hissd_learner"] = HISSDLearner

from .hissd_capability_team_learner import HISSDLearner as TeamAdapterLearner
REGISTRY["hissd_capability_team_learner"] = TeamAdapterLearner
