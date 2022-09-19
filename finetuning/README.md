# Finetuning 
In this folder we show how to fine-tune an autoregressive Language model on the following evaluation and downstream tasks with support for 7 programming languages:

* [APPS](https://huggingface.co/datasets/codeparrot/apps): Python benchmark to evaluate code generation. It is similar to HumanEval and MBPP, but it is more challanging and has more evaluation problems.
* [CodeComplex](https://huggingface.co/datasets/codeparrot/codecomplex): **Java** benchmark with a classification problem to predict the algorithmic complexity of Java programs among 7 labels.
* [CodeClone](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench): **Java** benchmark from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) dataset, with a binary classification problem of predicting the semantic equivalence of two programs. [WIP]
* [CodeDefect](https://huggingface.co/datasets/code_x_glue_cc_defect_detection): **C** benchmark from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE), with a binary classification problem of predicting whether a code is insecure code and may attack software systems. [WIP]
* [Code-to-text](https://huggingface.co/datasets/code_x_glue_ct_code_to_text): Dataset from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) for generationg natural language comments from code in **Python, Go, Java, Javascript, PHP and Ruby**. This task can also be done in a zero-shot setting without need for fine-tuning. [WIP]

We use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) API for all tasks, which supports distributed training on multiple GPUs. 

The evaluation score on the test set is shown at the end of the fine-tuning. For implementation details, please refer to the README inside each folder.
