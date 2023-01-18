build:
	docker build --network host --compress -t jodafons/rxpix2pixcycle:latest .
push:
	docker push jodafons/rxpix2pixcycle:latest
pull:
	singularity pull docker://jodafons/rxpix2pixcycle:latest
clean:
	docker system prune -a
	