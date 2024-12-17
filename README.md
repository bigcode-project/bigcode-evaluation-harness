## Setup

```bash
git clone https://github.com/IBM-OSS-Support/Granite-code-3.1-bigcode-evaluation-harness.git
cd Granite-code-3.1-bigcode-evaluation-harness
```
Install [`torch`](https://pytorch.org/get-started/locally/) based on your device type, and install the other packages using:
```
pip install -e .
```
To run the `DS-1000` benchmark, additional constraints must be resolved.
```
# python version must be 3.7.10
pip install -e ".[ds1000]" # installs all additional dependencies except PyTorch
# torch==1.12.1 required. Download version with relevant GPU support etc., e.g.,
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# to suppress any tensorflow optimization warnings, 
# precede call to "accelerate launch" with "TF_CPP_MIN_LOG_LEVEL=3"

# on some systems, tensorflow will attempt to allocate all GPU memory
# to its process at import which will raise a CUDA out-of-memory error
# setting "export TF_FORCE_GPU_ALLOW_GROWTH=true" resolves this
```
Also make sure you have `git-lfs` installed and are logged in the Hub
```
huggingface-cli login
````

We use [`accelerate`](https://huggingface.co/docs/accelerate/index) to generate code/text in parallel when multiple GPUs are present (multi-GPU mode). You can configure it using:

```bash
accelerate config
```

This evaluation harness can also be used in an evaluation only mode, you can use a Multi-CPU setting. For large models, we recommend specifying the precision of the model using the `--precision` flag instead of accelerate config to have only one copy of the model in memory. You can also load models in 8bit with the flag `--load_in_8bit` or 4bit with `--load_in_4bit` if you have `bitsandbytes` installed with the required transformers and accelerate versions.

# New Evaluation Tasks for Granite-3.1 Models

This section describes five new tasks added to the bigcode-evaluation-harness for evaluating code generation capabilities, specifically tested with IBM Granite 3.1 model.

## Tasks Overview

### 1. Code Porting (code-porting)
Tests the model's ability to port code between programming languages while maintaining functionality.
- Currently supports Python to Java conversion
- Evaluates structural correctness and implementation details
- Checks for proper imports, class structure, and method signatures

```bash
accelerate launch main.py \
  --model YOUR_MODEL \
  --tasks code-porting \
  --temperature 0.001 \
  --n_samples 1 \
  --batch_size 1 \
  --top_k 40 \
  --top_p 0.9 \
  --allow_code_execution \
  --max_length_generation 2048
  ```



### Evaluation Metric
- Similarity score
- Import verification
- Structural correctness
- Implementation accuracy

### 2. Quarkus Refactoring (quarkus-refactoring)
Evaluates the ability to refactor traditional Java code into Quarkus-based REST applications.
- Tests proper REST endpoint creation
- Validates appropriate HTTP methods and status codes
- Checks for proper annotations and dependency injection

```bash
accelerate launch main.py \
  --model YOUR_MODEL \
  --tasks quarkus-refactoring \
  --temperature 0.001 \
  --n_samples 1 \
  --batch_size 1 \
  --top_k 40 \
  --top_p 0.9 \
  --allow_code_execution \
  --max_length_generation 2048
  ```
### Evaluation Metric
- REST endpoint coverage
- Annotation correctness
- HTTP method implementation
- Response handling

### 3. Unit Test Generation (unittest-generation)
Tests the model's ability to generate comprehensive unit tests for Python code.
- Validates test case coverage
- Checks for proper test setup and teardown
- Ensures proper assertions and error handling

```bash
accelerate launch main.py \
  --model YOUR_MODEL \
  --tasks unittest-generation \
  --temperature 0.001 \
  --n_samples 1 \
  --batch_size 1 \
  --top_k 40 \
  --top_p 0.9 \
  --allow_code_execution \
  --max_length_generation 2048
  ```
### Evaluation Metric
- Test case coverage
- Test structure completeness
- Assertion correctness
- Error handling coverage


### 4. Documentation Generation (documentation-generation)
Evaluates the ability to generate comprehensive documentation for Python code.
- Tests class-level and method-level documentation
- Checks for proper docstring format
- Validates parameter and return value documentation

```bash
accelerate launch main.py \
  --model YOUR_MODEL \
  --tasks documentation-generation \
  --temperature 0.001 \
  --n_samples 1 \
  --batch_size 1 \
  --top_k 40 \
  --top_p 0.9 \
  --allow_code_execution \
  --max_length_generation 2048
  ```
### Evaluation Metric
- Class documentation score
- Method documentation coverage
- Parameter documentation
- Format adherence

### 5. Bug Fixing (bug-fixing)
Tests the model's ability to identify and fix bugs in Python code.
- Evaluates syntax error detection and correction
- Checks logical error fixes
- Validates code structure maintenance

```bash
accelerate launch main.py \
  --model YOUR_MODEL \
  --tasks bug-fixing \
  --temperature 0.001 \
  --n_samples 1 \
  --batch_size 1 \
  --top_k 40 \
  --top_p 0.9 \
  --allow_code_execution \
  --max_length_generation 2048
  ```
### Evaluation Metric
- Syntax fix score
- Code structure maintenance
- Functionality verification

## Output Format
Each task generates a JSON output with:
- Task-specific scores
- Detailed breakdown of evaluation metrics
- Error messages (if any)

```bash
{
  "task_name": {
    "score": 85.5,
    "details": {
      "category1": "8/10 (80%)",
      "category2": "9/10 (90%)"
    }
  }
}
  ```
