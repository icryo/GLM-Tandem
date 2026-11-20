# Incremental tokenization must equal a full encode (CPU only; MODEL_PATH=/path/to/checkpoint): growing, edited, interleaved conversations with tools, reasoning, unicode.
import os, random, sys, time
from pathlib import Path
from mlx_lm.utils import load_tokenizer
src = open(os.environ.get("SERVER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "tpserve.py"))).read(); a = src.index("if os.environ.get(\"TP_TOKMEMO\", \"1\") == \"1\":"); b = src.index("# ---- Never hang")
exec(src[a:b])
tok = load_tokenizer(Path(os.environ["MODEL_PATH"]))            # the checkpoint directory; hf = tok._tokenizer
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(int(os.environ.get("SEED", "1")))
EXTRA = ["<|user|>", "<think>", "</think>", "<|assistant|>\n", "naïve café 日本語 🙂", "\n\n", "  ", "<tool_call>", "{\"a\": [1, 2]}", "<|observation|>", "\t", "<|endoftext|>", "[gMASK]", " <|user|> "]
def words(n): return " ".join(rnd.choice(WORDS) if rnd.random() > 0.02 else rnd.choice(EXTRA) for _ in range(n))
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
def render(msgs, **kw): return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
convs = []; bad = 0; n = 0; t_inc = t_full = 0
for c in range(4):
    convs.append(([{"role": "system", "content": "Sys " + words(rnd.choice([50, 4000, 15000]))}], dict(tools=tools) if c % 2 else {}))
for step in range(int(os.environ.get("STEPS", "60"))):
    msgs, kw = rnd.choice(convs)
    op = rnd.random()
    if op < 0.1 and len(msgs) > 4:                      # client edits history
        k = rnd.randrange(1, len(msgs)); msgs[k] = dict(msgs[k], content=words(rnd.randrange(1, 300)))
    elif op < 0.15 and len(msgs) > 6: del msgs[2:4]     # client drops old turns
    if msgs[-1]["role"] != "user": msgs.append({"role": "user", "content": words(rnd.randrange(1, 900))})
    t = time.perf_counter(); got = render(msgs, **kw); t_inc += time.perf_counter() - t
    t = time.perf_counter(); text = _tw_act(tok, msgs, add_generation_prompt=True, tokenize=False, **kw); want = hf.encode(text, add_special_tokens=False); t_full += time.perf_counter() - t
    n += 1; bad += list(got) != list(want)
    sysn = render(msgs[:1] + [{"role": "user", "content": ""}], **kw); bad += list(sysn) != list(hf.encode(_tw_act(tok, msgs[:1] + [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False))
    if kw and rnd.random() < 0.5: msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": words(2)}}}]}); msgs.append({"role": "tool", "tool_call_id": "c1", "content": words(rnd.randrange(1, 1200))})
    else: msgs.append({"role": "assistant", "content": words(rnd.randrange(1, 200)), **({"reasoning_content": words(30)} if rnd.random() < 0.3 else {})})
print(f"{n} renders, mismatches {bad}, incremental {1e3 * t_inc / n:.1f} ms vs full {1e3 * t_full / n:.1f} ms per render, longest {max(len(e[1]) for e in _TM)} tokens, entries {len(_TM)}")
