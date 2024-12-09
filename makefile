# There are two dockerfiles: for all benchmarks, and for MultiPL-E
DOCKERFILE=Dockerfile

ifeq ($(DOCKERFILE), Dockerfile)
	IMAGE_NAME=evaluation-harness
else
	IMAGE_NAME=evaluation-harness-multiple
endif

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

test:
	docker run -v $(CURDIR)/tests/docker_test/test_generations.json:/app/test_generations.json:ro \
	-it $(IMAGE_NAME) python3 main.py --model dummy_model --tasks humaneval --limit 4 \
	--load_generations_path /app/test_generations.json --allow_code_execution 

	@echo "If pass@1 is 0.25 then your configuration for standard benchmarks is correct"

all: build test