import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from utils import form_llm_input, test_model_output, generate_and_get_logits
from typing import Dict
from repeng import ControlModel, ControlVector, DatasetEntry

device = "cuda:0"
model_id = "../models/StruQ/models/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
model = ControlModel(model, list(range(-5, -18, -1)))

"""
Need a dataset to generate the control vectors.
"""

