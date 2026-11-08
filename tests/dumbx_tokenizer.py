"""A small fast tokenizer built as yhavinga/dutch-llama-tokenizer is: no pre-tokenizer, a normalizer that prepends ▁ and
turns spaces into ▁, and the templates <s> A and <s> A <s> B. So word_ids, offsets and overflowing windows behave as
with the real one, offline. Its pieces: specials, ▁, single characters and some whole and partial words; the Unigram
model takes the fewest pieces."""
import string

from tokenizers import Tokenizer, normalizers
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
PIECES = ["▁de", "▁het", "▁een", "▁man", "▁vrouw", "▁loopt", "▁zit", "▁in", "▁kamer", "▁Jan", "▁Piet", "▁woont", "▁Amsterdam",
          "▁Wie", "▁waar", "▁is", "▁vraag", "▁bij", "▁ijz", "ige", "▁kalm", "te", "▁Hertogen", "bosch", "▁zag", "▁mezelf",
          "▁wij", "▁ik", "▁bekijk", "▁bekijken", "en", "▁foto", "▁van"]


def make_tokenizer():
    characters = list(string.ascii_letters + string.digits + string.punctuation)
    vocab = [(token, 0.0) for token in SPECIALS] + [(piece, -1.0) for piece in ["▁"] + characters + PIECES]
    backend = Tokenizer(Unigram(vocab, unk_id=0))
    backend.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.post_processor = TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")
