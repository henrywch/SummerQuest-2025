# eval_config.py

from opencompass.models import HuggingFaceCausalLM, VLLM
from opencompass.datasets import GSM8KDataset

from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate

# Reader 只需指定输入列和输出列
gsm8k_reader = dict(
    input_columns=['question'],
    output_column='answer'
)

models = [
    # 教师模型（vLLM 后端，多卡）
    dict(
        type=VLLM,
        abbr='qwen2.5-math-7b',
        path="/data-mnt/data/chwang/models/Qwen2.5-Math-7B",
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2)
    ),
    # 原始小模型 (Transformer 后端，单卡)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-raw',
        path="/data-mnt/data/chwang/models/Qwen2.5-0.5B",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    ),
    # SFT 后小模型 (Transformer 后端，单卡)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-sft',
        path="LLaMA-Factory/saves/qwen2.5-0.5b/lora/sft",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]

datasets = [
  dict(
    type=GSM8KDataset,
    abbr='gsm8k',
    path='opencompass/gsm8k',
    reader_cfg=gsm8k_reader,
    infer_cfg=dict(
      retriever=dict(type='ZeroRetriever'),
      inferencer=dict(type='GenInferencer'),
      prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role="HUMAN", prompt="{question}")])
      ),
    ),
    eval_cfg=dict(
      evaluator=dict(type='AccEvaluator'),
      pred_postprocessor=dict(type='first_capital_postprocess'),
    )
  )
]
