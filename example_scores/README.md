
# Scores

For the experiments below, we use the default setting of top-p sampling with `p=0.95` and adapt the temperature.

## MBPP
For MBPP we use temperature 0.1:

<div align="center">
	
|Task | Model  | pass@1 | 
|-------|--------|---------|
|MBPP | InCoder (1B) | 10.6% | 
|MBPP | CodeParrot (1.5B) | 0.2%(to be updated) |
|MBPP | BigCode-any-license (340M) | 17% |
|MBPP | BigCode-safe-license (340M) | 10.2% |
|MBPP | CodeGen-Mono (16B) | 42.4%(*) |
|MBPP | code-cushman-001/davinci-002 (Codex) | 45%/58%(*)(+) |	
</div>

(*) score reported on [CodeT](https://arxiv.org/pdf/2207.10397v1.pdf) paper

(+) these models are variants of Codex available thrigh OpenAI API, size is unknown

## APPS
For APPS we use temperature 0.1:

   * Average accuracy:

<div align="center">
	
|Task | Model | Introductory| Interview| Competition| Average (weighted)|
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) | 10.01%| 7.52% | 4.29% | 7.37%|
|APPS | BigCode all license(340M) | **14.95%** | **10.47%**| **8.43%**| **10.95%**|
|APPS | BigCode safe license(340M) | 13.71% | 9.60% | 5.99%|9.7% |
	
</div>

* Strict accuracy:
<div align="center">

|Task | Model  | Introductory | Interview| Competition| Average (weighted)|
|-------|--------|--------|-------|-------|-------|
|APPS | GPT2 finetuned (1.5B) |2.4%| 0.5% | 0% |0.78% |
|APPS | BigCode all license(340M) | 3.4% | **0.7%**| **0.7%**|**1.24%**|
|APPS | BigCode safe license(340M) | **3.8%** |**0.7%** | 0.2%| 1.22%|
	
</div>

* Pass@k scores:

## CoNaLa (python)
With temperature 0.1:

|Task | Model  |BLEU| 
|-------|--------|---------|
|CoNaLa | InCoder (1B) | 17.77% | 
|CoNaLa | BigCode safe license (340M) | 9.89% | 
|CoNaLa | BigCode all license (340M) | 12.33% | 

## Spider (sql)
With temperature 0.1:

|Task | Model  |BLEU| 
|-------|--------|---------|
|Spider | InCoder (1B) | 5.91% | 

## Concode (java)
With temperature 0.1 (on the first 1200 samples from the test set):

|Task | Model  |BLEU| 
|-------|--------|---------|
|Spider | InCoder (1B) | 2.34% | 

## Evaluation time on 8 A100 GPUs:
- Evaluation on MBPP is 1-shot for 500 prompts, the evaluation takes **~1 hour**
- Evaluation on APPS (total of 5000 prompts) with single generations to compute average accuracy/strict accuracy takes in average **~4 hours for each of the 3 difficulty levels**
- The evaluation on APPS with multiple generations (nsamples=200) to compute pass@k takes **~16 hours for each 1000 samples (introductory level for example)**
- Th evaluation on CoNala and Spider is fast and takes a couple of minutes, even with 1 GPU. ConCode takes less than 2h on 4 A100.
