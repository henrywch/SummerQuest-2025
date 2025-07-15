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
from opencompass.utils.text_postprocessors import first_capital_postprocess  # Fixed import

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
        run_cfg=dict(num_gpus=1)
    ),
    # SFT small model (HuggingFace backend, single GPU)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-sft',
        path="/data-mnt/data/chwang/models/Qwen2.5-0.5B-LoRA",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]

# ============================= Dataset Config =============================
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
                    dict(role="HUMAN", prompt="{question}")
                ])
            )
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
            # Fixed: Use imported postprocessor function
            pred_postprocessor=dict(type=first_capital_postprocess),
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
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
work_dir = osp.join('outputs', f'gsm8k_comparison_{timestamp}')

# ============================ Summarizer Config ==========================
summarizer = dict(
    type="SingleTableSummarizer",
    dataset_abbrs=['gsm8k'],
    summary_groups=[{
        'name': 'gsm8k_accuracy',
        'subsets': ['gsm8k'],
        'metrics': ['accuracy']
    }]
)