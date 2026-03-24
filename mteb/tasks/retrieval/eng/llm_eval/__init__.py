from mteb.tasks.retrieval.eng.llm_eval.llm_aila_statutes import LLMAILAStatutes
from mteb.tasks.retrieval.eng.llm_eval.llm_legal_bench_corporate_lobbying import (
    LLMLegalBenchCorporateLobbying,
)
from mteb.tasks.retrieval.eng.llm_eval.llm_spartqa import LLMSpartQA
from mteb.tasks.retrieval.eng.llm_eval.llm_tempreason_l1 import LLMTempReasonL1
from mteb.tasks.retrieval.eng.llm_eval.llm_twitter_hjerne import (
    LLMTwitterHjerneRetrieval,
)
from mteb.tasks.retrieval.eng.llm_eval.llm_winogrande import LLMWinoGrande

__all__ = [
    "LLMTempReasonL1",
    "LLMLegalBenchCorporateLobbying",
    "LLMAILAStatutes",
    "LLMSpartQA",
    "LLMWinoGrande",
    "LLMTwitterHjerneRetrieval",
]
