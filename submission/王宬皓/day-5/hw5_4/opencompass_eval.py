# eval_config.py
import os.path as osp
from opencompass.models import HuggingFaceCausalLM, VLLM
from opencompass.datasets import GSM8KDataset
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.runners import LocalRunner
from opencompass.partitioners import NumWorkerPartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.utils.text_postprocessors import gsm8k_postprocess

# Import for timestamp
from datetime import datetime

# ============================== Model Configs ==============================
models = [
    # Teacher model (vLLM backend, multi-GPU)
    dict(
        type=VLLM,
        abbr='qwen2.5-math-7b',
        path="/data-mnt/data/chwang/models/Qwen2.5-Math-7B",
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2)
    ),
    # Raw small model (HuggingFace backend, single GPU)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-raw',
        path="/data-mnt/data/chwang/models/Qwen2.5-0.5B",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        generation_kwargs={
            'eos_token_id': 151643,  # Qwen tokenizer eos_token_id
            'pad_token_id': 151643,
        }
    ),
    # SFT small model (HuggingFace backend, single GPU)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-sft',
        path="/data-mnt/data/chwang/models/Qwen2.5-0.5B-LoRA",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        generation_kwargs={
            'eos_token_id': 151643,  # Qwen tokenizer eos_token_id
            'pad_token_id': 151643,
        }
    )
]

# ============================= Dataset Config =============================
# Proper GSM8K prompt template
gsm8k_prompt = """
Solve the following math problem step by step. Put your final answer in a boxed format.

Question: {question}"""

# Reader configuration
gsm8k_reader = dict(
    input_columns=['question'],
    output_column='answer'
)

# Dataset configuration
datasets = [
    dict(
        type=GSM8KDataset,
        abbr='gsm8k',
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader,
        infer_cfg=dict(
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role="HUMAN", prompt=gsm8k_prompt.strip())
                ])
            )
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
            pred_postprocessor=dict(type=gsm8k_postprocess),  # Correct GSM8K processor
        )
    )
]

# ========================== Parallel Evaluation ==========================
# Inference configuration (parallel execution)
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,  # Distributes tasks across workers
        num_worker=min(8, len(models))  # Dynamic worker count
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask)  # Inference task type
    )
)

# Evaluation configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner,      # Simple partitioner for evaluation
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask)  # Evaluation task type
    )
)

# ============================ Output Settings ============================
# Set output directory with timestamp to avoid conflicts
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
work_dir = osp.join('outputs', f'gsm8k_comparison_{timestamp}')

# ============================ Summarizer Config ==========================
summarizer = dict(
    type="ComparisonSummarizer",  # Changed to comparison type
    dataset_abbrs=['gsm8k'],
    summary_groups=[{
        'name': 'gsm8k_comparison',
        'subsets': ['gsm8k'],
        'models': [
            'qwen2.5-math-7b', 
            'qwen2.5-0.5b-raw', 
            'qwen2.5-0.5b-sft'
        ],  # Explicitly list models for comparison
        'metrics': ['accuracy']
    }]
)