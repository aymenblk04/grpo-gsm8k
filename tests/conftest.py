"""Tiny randomly-initialised model + tokenizer so tests run offline on CPU."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["<think> 2 + 2 = 4 </think> <answer> 4 </answer> What is the answer?"] * 50
    specials = ["<unk>", "<pad>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=specials))
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>"
    )
    hf.chat_template = CHAT_TEMPLATE
    return hf


@pytest.fixture(scope="session")
def tiny_model_path(tiny_tokenizer, tmp_path_factory):
    """Saved to disk because GRPOTrainer reloads the reference model by path."""
    config = LlamaConfig(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tiny_tokenizer.pad_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny-llama")
    LlamaForCausalLM(config).save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return str(path)
