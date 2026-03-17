# 05 Generation Code

This directory contains the classroom demo scripts for the text generation lecture.

## Recommended teaching order

1. `01_compare_decoding_strategies.py`
   Use this first when you cover greedy decoding, beam search, and basic sampling.

2. `02_analyze_generation_scores.py`
   Use this when you cover log-probabilities, evaluation, and perplexity.

3. `03_temperature_topk_topp_lab.py`
   Use this for the temperature, top-k, top-p, and lab sections.

4. `04_modern_decoding_methods.py`
   Use this for contrastive search, repetition penalty, and typical sampling.

5. `05_assisted_generation_demo.py`
   Use this for the speculative decoding / assisted generation slide.

## How to run

From this directory:

```bash
pipenv install
pipenv shell
```

Then run any script with Python:

```bash
python 01_compare_decoding_strategies.py --prompt "The Catholic University of America is a" --model gpt2
python 02_analyze_generation_scores.py --prompt "Explain entropy in plain English in one short paragraph for undergraduate students:" --model gpt2
python 03_temperature_topk_topp_lab.py --prompt "Write one paragraph about the moon:" --model gpt2
python 04_modern_decoding_methods.py --prompt "The future of AI in education is" --model gpt2
python 05_assisted_generation_demo.py --prompt "The Catholic University of America is preparing students for a future where AI can support learning, research, and advising. " --target_model facebook/opt-1.3b --assistant_model facebook/opt-350m --max_new_tokens 256 --no_repeat_ngram_size 0 --repetition_penalty 1.0
```

You can also run them directly with `pipenv run`:

```bash
pipenv run python 01_compare_decoding_strategies.py --prompt "Write a short paragraph about Mars:" --model gpt2
```

## Three model options

These are the three best model choices for this lecture, depending on your hardware:

1. `gpt2`
   Best default for classroom demos. Small, fast, and reliable enough to show decoding differences clearly.

2. `gpt2-medium`
   Good middle option when you want somewhat better fluency without a large hardware jump.

3. `mistralai/Mistral-7B-Instruct-v0.2`
   Best stronger modern option if you have a good GPU or a well-configured lab machine.

## Assisted generation recommendations

For the speculative decoding / assisted generation demo, use a separate recommendation from the general lecture defaults.

Best positive example found for this setup:

```bash
python 05_assisted_generation_demo.py --prompt "The Catholic University of America is preparing students for a future where AI can support learning, research, and advising. " --target_model facebook/opt-1.3b --assistant_model facebook/opt-350m --max_new_tokens 256 --no_repeat_ngram_size 0 --repetition_penalty 1.0
```

Notes:

- This pair produced a small but positive speedup on the current machine.
- Some GPT-2 family pairs worked inconsistently and were often slower in assisted mode.
- For speed-focused demos, setting `--no_repeat_ngram_size 0 --repetition_penalty 1.0` avoids extra logits-processor overhead.
- Assisted generation is a systems optimization, so whether it helps depends on the model pair, hardware, and runtime overhead.

## Which script matches which slide section

- `01_compare_decoding_strategies.py`
  Matches the slides on greedy search, beam search, sampling, and the generation pipeline.

- `02_analyze_generation_scores.py`
  Matches the slides on log probabilities, log-probability versus quality, evaluation, and perplexity.

- `03_temperature_topk_topp_lab.py`
  Matches the slides on temperature, top-k sampling, top-p sampling, and the lab.

- `04_modern_decoding_methods.py`
  Matches the slides on contrastive search, repetition penalty, typical sampling, and modern decoding methods.

- `05_assisted_generation_demo.py`
  Matches the slide on speculative decoding / assisted generation.
