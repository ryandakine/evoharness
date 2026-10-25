"""evoharness: a small harness for building, running and optimizing LLM agent graphs."""
from .attribution import Attribution, dag_shapley, shapley, viable
from .distribution import DistributionConfig, DistributionResult, GraphDistribution, reinforce_graph
from .evaluation import HeldOutScore, TaskSplit, evaluate_frozen, split_tasks
from .executor import AgentError, RunResult, Task, exact_match, extract_answer, run_genome
from .fitness import Evaluator, majority_vote, single_agent
from .ga import GAConfig, GAResult, evolve
from .genome import DEFAULT_ROLES, Genome
from .llm import (
    APIBackend,
    Backend,
    BudgetExceeded,
    CachedBackend,
    EmptyResponse,
    MockBackend,
    ModelSpec,
    Response,
    is_transient,
)
from .online import Experience, PromptUpdate, optimize_prompts
from .swarm import ParticleSwarm, PSOConfig, RoleSearchResult, g_decode, jfk_scores, search_roles

__version__ = "0.2.0"

__all__ = [
    "APIBackend", "AgentError", "Attribution", "Backend", "BudgetExceeded", "CachedBackend",
    "DEFAULT_ROLES", "DistributionConfig", "DistributionResult", "EmptyResponse", "Evaluator",
    "Experience", "GAConfig", "GAResult", "Genome", "GraphDistribution", "HeldOutScore",
    "MockBackend", "ModelSpec", "PSOConfig", "ParticleSwarm", "PromptUpdate", "Response",
    "RoleSearchResult", "RunResult", "Task", "TaskSplit", "dag_shapley", "evaluate_frozen",
    "evolve", "exact_match", "extract_answer", "g_decode", "is_transient", "jfk_scores",
    "majority_vote", "optimize_prompts", "reinforce_graph", "run_genome", "search_roles",
    "shapley", "single_agent", "split_tasks", "viable", "__version__",
]
