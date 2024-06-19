
all: build_local
SHELL := /bin/bash



build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

build_images:
	make build_base
	make build_sif


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/p2p-cycle:base -f Dockerfile .


build_sif:
	docker push ${DOCKER_NAMESPACE}/p2p-cycle:base
	singularity pull docker://${DOCKER_NAMESPACE}/p2p-cycle:base
	mv *.sif ${PROJECT_DIR}/images



run:
	singularity run --nv --bind=/home:/home  --bind=${PROJECT_DIR}:${PROJECT_DIR} --writable-tmpfs ${PROJECT_DIR}/images/p2p-cycle_base.sif  

run_jupyter:
	singularity exec --nv --bind=/home:/home --bind=${PROJECT_DIR}:${PROJECT_DIR} --writable-tmpfs ${PROJECT_DIR}/images/p2p-cycle_base.sif /bin/bash ./activate.sh jupyter


clean:
	docker system prune -a
	