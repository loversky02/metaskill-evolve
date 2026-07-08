"""MetaSkill-Evolve — a $0/Mac reproduction and dissection of arXiv:2607.05297."""
from . import agents
from .benchmarks import ALFWorldTask, MockAlfEnv, QATask, load_gsm8k, load_sealqa
from .config import BACKBONES, STRENGTH_ORDER, BackboneSpec, Config, backbone, faithful
from .dag import DAG, Node
from .evolve import EvolveState, run_evolution
from .fast_loop import FastState, fast_iteration, run_fast
from .llm import LLM, CountingLLM, MLXClient, MockLLM, OpenAICompatClient, make_llm
from .retriever_dpp import dpp_retrieve
from .skills import META_COMPONENTS, SEED_META, SEED_TASK, SkillStore
from .slow_loop import meta_productivity, slow_iteration
from .synthetic import LetterCountTask
from .tasks import RuleTask, Task, evaluate, worst_example

__version__ = "0.1.0"

__all__ = [
    "BackboneSpec", "Config", "faithful", "backbone", "BACKBONES", "STRENGTH_ORDER",
    "DAG", "Node",
    "LLM", "MockLLM", "OpenAICompatClient", "MLXClient", "make_llm", "CountingLLM",
    "dpp_retrieve",
    "SkillStore", "META_COMPONENTS", "SEED_META", "SEED_TASK",
    "Task", "RuleTask", "LetterCountTask", "evaluate", "worst_example",
    "agents", "run_fast", "fast_iteration", "FastState",
    "run_evolution", "EvolveState", "slow_iteration", "meta_productivity",
    "QATask", "ALFWorldTask", "MockAlfEnv", "load_gsm8k", "load_sealqa",
]
