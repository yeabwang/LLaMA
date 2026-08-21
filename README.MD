# LLaMA 2

![1786945099940](llama_arc.png)

## ROPE

![1787023316542](rope.png)

## Grouped Query Attention

It's a good compromise between quality and speed between the Multi Head Attention and the Multi Query Attention.

## Inference

```bash
python inference.py --prompt "Who wrote the Odyssey?"
```

Flags: `--max-gen-len` (64), `--temperature` (0.6), `--top-p` (0.9),
`--max-seq-len` (1024), `--allow-cuda`, `--checkpoints-dir`, `--tokenizer-path`.

The prompt is passed through verbatim — no template is applied. Prompt shaping
is done in the string you pass. Forms worth testing:

**Plain continuation** — raw language modeling, no scaffolding.

```bash
python inference.py --prompt "Simply put, the theory of relativity states that"
```

**Instruction (Llama-2 chat format)** — for chat-tuned checkpoints.

```bash
python inference.py --prompt "[INST] Explain gradient descent in two sentences. [/INST]"
```

**Few-shot** — a couple of exemplars pin the output format.

```bash
python inference.py --prompt "Q: What is the capital of France?
A: Paris.

Q: Who wrote Hamlet?
A: William Shakespeare.

Q: Who wrote the Odyssey?
A:"
```

**Zero-shot chain of thought** — the trailing cue elicits reasoning steps.

```bash
python inference.py --prompt "Q: A shop sells pens at 3 for \$2. How much do 12 pens cost?
A: Let's think step by step."
```

Multi-line prompts work as shown in bash/zsh. In PowerShell use a here-string:

```powershell
python inference.py --prompt @'
Q: What is the capital of France?
A: Paris.

Q: Who wrote the Odyssey?
A:
'@
```
