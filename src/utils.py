from ..models.StruQ.config import PROMPT_FORMAT, DELIMITERS, DEFAULT_TOKENS

import functools
import gc
import inspect
import torch
from torch import Tensor
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def get_not_allowed_ids(tokenizer, device, allow_non_ascii, not_allowed_tokens):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    not_allowed_tokens = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            not_allowed_tokens.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    # ADD other tokens that we cannot sample (delimiters!)

    return torch.tensor(nonascii_toks, device=device)

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

def form_llm_input(data):
    """
    format of data: map with keys: 'instruction', 'input', 'generator', 'output'
    """
    frontend_delimiters = "SpclSpclSpcl"
    prompt_format = PROMPT_FORMAT[frontend_delimiters]
    llm_input = prompt_format['prompt_input'].format_map(data)

    return llm_input

def test_model_output(llm_input, model, tokenizer):
    model.generation_config.max_new_tokens = 512
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0


    input_ids = tokenizer.encode(llm_input,
                                 return_tensors="pt",
                                 padding="longest",
                                 max_length=tokenizer.model_max_length,
                                 truncation=True)

    output_ids = model.generate(input_ids.to(model.device),
                                      attention_mask=torch.ones_like(input_ids).to(model.device),
                                      generation_config=model.generation_config,
                                      pad_token_id=tokenizer.pad_token_id)[0][input_ids.shape[1]:]

    outp = tokenizer.decode(output_ids, skip_special_tokens=False)

    start = 0
    while outp[start] == ' ': start += 1
    outp = outp[start:outp.find(DEFAULT_TOKENS['eos_token'])]

    return outp

def generate_and_get_logits(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=500,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    logits = output.scores
    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
    return logits, generated_text
