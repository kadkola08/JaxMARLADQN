NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
BASE_FLAGS=-it --rm -v $(CURDIR):/home/$(MYUSER) --shm-size 20G
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = jaxmarl
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
# Real invoking user when make is run via sudo (id -u alone is 0).
ID = $(if $(SUDO_UID),$(SUDO_UID),$(shell id -u))

# Build context is this directory (JaxMARL/). If JaxMARL has no .git (e.g. marlhf
# monorepo), the Dockerfile clones JaxRobotarium instead of `git submodule update`.
# make file commands
build:
	sudo DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain $(CURDIR)/.

run:
	sudo $(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

workflow-test:
	# without -it flag
	docker run --rm -v $(CURDIR):/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"

