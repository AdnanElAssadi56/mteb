from mteb.tasks.retrieval.eng.llm_eval.llm_aila_statutes import LLMAILAStatutes
from mteb.tasks.retrieval.eng.llm_eval.llm_fquad import LLMFQuADRetrieval
from mteb.tasks.retrieval.eng.llm_eval.llm_hc3_finance import (
    LLMHC3FinanceRetrieval,
)
from mteb.tasks.retrieval.eng.llm_eval.llm_legal_bench_corporate_lobbying import (
    LLMLegalBenchCorporateLobbying,
)
from mteb.tasks.retrieval.eng.llm_eval.llm_legalbench_consumer_contracts_qa import (
    LLMLegalBenchConsumerContractsQA,
)
from mteb.tasks.retrieval.eng.llm_eval.llm_public_health_qa import (
    LLMPublicHealthQA,
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
    # V3 — post-audit additions (active in paper)
    "LLMFQuADRetrieval",
    "LLMLegalBenchConsumerContractsQA",
    "LLMPublicHealthQA",
    "LLMHC3FinanceRetrieval",
]
