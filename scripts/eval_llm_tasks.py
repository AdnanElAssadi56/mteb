import mteb
import torch
import torch._dynamo
import gc

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from mteb.tasks.llm_eval.classification.eng import *
from mteb.tasks.llm_eval.classification.multilingual import *
from mteb.tasks.llm_eval.sts.eng import *
from mteb.tasks.llm_eval.sts.multilingual import *
from mteb.tasks.llm_eval.clustering.eng import *
from mteb.tasks.llm_eval.pair_classification.eng import *
from mteb.tasks.llm_eval.pair_classification.multilingual import *
from mteb.tasks.retrieval.eng.llm_eval import *

tasks = [
    # LLMBanking77Classification(),
    LLMImdbClassification(),
    # LLMToxicConversationsClassification(),
    # LLMTweetSentimentExtractionClassification(),
    # LLMAmazonCounterfactualClassification(),
    # LLMMassiveIntentClassification(),
    # LLMMassiveScenarioClassification(),
    # LLMMTOPDomainClassification(),

    # LLMSTSBenchmark(),
    # LLMSTS12(),
    # LLMSTS13(),
    # LLMSTS14(),
    # LLMSTS15(),
    # LLMSTS16(),
    # LLMBIOSSES(),
    # LLMSICKR(),
    # LLMSTS17(),
    # LLMSTS22v2(),

    # LLMArxivClusteringP2P(),
    # LLMArxivClusteringS2S(),
    # LLMBigPatentClustering(),
    # LLMBiorxivClusteringP2PV2(),
    # LLMMedrxivClusteringP2PV2(),
    # LLMMedrxivClusteringS2SV2(),
    # LLMRedditClusteringP2P(),
    # LLMStackExchangeClusteringP2PV2(),
    # LLMStackExchangeClusteringV2(),
    # LLMTwentyNewsgroupsClusteringV2(),

    # LLMSprintDuplicateQuestionsPC(),
    # LLMLegalBenchPC(),
    # LLMTwitterURLCorpusPC(),
    # LLMRTE3PC(),

    # LLMTempReasonL1(),
    # LLMAILAStatutes(),
    # LLMLegalBenchCorporateLobbying(),
    # LLMSpartQA(),
    # LLMTwitterHjerneRetrieval(),
    # LLMWinoGrande(),
]
tasks[0].n_experiments = 1

MODELS = [
    # Top 3 (already complete)
    "tencent/KaLM-Embedding-Gemma3-12B-2511",  # Rank #1 ✓
    # #"nvidia/llama-embed-nemotron-8b",          # Rank #2 ✓
    # "Qwen/Qwen3-Embedding-8B",                 # Rank #3 ✓

    # # 4B models
    # "Qwen/Qwen3-Embedding-4B",                 # Rank #5 ✓
    # "codefuse-ai/F2LLM-v2-4B",                 # Rank #10 NEW

    # # 0.6B-1.7B models
    # "Qwen/Qwen3-Embedding-0.6B",               # Rank #13 ✓
    # "codefuse-ai/F2LLM-v2-1.7B",               # Rank #12 ✓

    # # 7-8B models
    # "bflhc/Octen-Embedding-8B",                # Rank #6 NEW
    # "codefuse-ai/F2LLM-v2-8B",                 # Rank #8 NEW
    # "Alibaba-NLP/gte-Qwen2-7B-instruct",       # Rank #15 ✓
    # "GritLM/GritLM-7B",                        # Rank #23 ✓
    # "intfloat/e5-mistral-7b-instruct",         # Rank #25 ✓
    # "Linq-AI-Research/Linq-Embed-Mistral",     # Rank #16 NEW

    # # 14B+ models
    # "codefuse-ai/F2LLM-v2-14B",                # Rank #7 NEW

    # # Jina v5 (much better than v3!)
    # "jinaai/jina-embeddings-v5-text-small",    # Rank #11 NEW
    # "jinaai/jina-embeddings-v5-text-nano",     # Rank #14 NEW

    # # Additional top models (17-35)
    # "intfloat/multilingual-e5-large-instruct", # Rank #17 NEW
    # "codefuse-ai/F2LLM-v2-0.6B",               # Rank #18 NEW
    # "google/embeddinggemma-300m",              # Rank #20 NEW (local, multilingual)
    # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",     # Rank #27 NEW
    # "Salesforce/SFR-Embedding-2_R",            # Rank #29 NEW             # Rank #30 NE

    # Important baselines (general + multilingual)
    # "BAAI/bge-m3",                             # Rank #39 NEW (multilingual)
    # "intfloat/multilingual-e5-large",          # Rank #40 NEW (multilingual)
    # "Snowflake/snowflake-arctic-embed-l-v2.0", # Rank #50 NEW
    # "intfloat/multilingual-e5-base",           # Rank #53 NEW (multilingual)
    # "intfloat/multilingual-e5-small",          # Rank #55 NEW (multilingual)

]

MODELS = [
    # ("tencent/KaLM-Embedding-Gemma3-12B-2511", 32),   # was 216, OOM
    # ("Qwen/Qwen3-Embedding-8B",                 64),   # was 503
    ("jinaai/jina-embeddings-v5-text-small",     256),  # was 935
    # ("Qwen/Qwen3-Embedding-0.6B",               128),  # was 1089
    # ("jinaai/jina-embeddings-v5-text-nano",      1024), # was 1945
    # ("intfloat/multilingual-e5-small",           1024), # was 1945
]

cache = mteb.ResultCache(cache_path="throughput_results")

for model_name, bs in MODELS:
    print(f"\n{'='*80}\nEvaluating: {model_name}\n{'='*80}")

    model = mteb.get_model(
        model_name,
    )

    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        encode_kwargs={"batch_size": bs}, 
        raise_error=False,
    )

    print(f"Completed: {model_name}\n")
