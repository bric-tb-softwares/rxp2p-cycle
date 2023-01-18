build:
	docker build --network host --compress -t jodafons/rxp2p-cycle:latest .
push:
	docker push jodafons/rxp2p-cycle:latest
pull:
	singularity pull docker://jodafons/rxp2p-cycle:latest
clean:
	docker system prune -a
	