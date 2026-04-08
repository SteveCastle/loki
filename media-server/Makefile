IMAGE_NAME  ?= lowkey-media-server
IMAGE_TAG   ?= latest
CONTAINER   ?= lowkey-media-server
PORT        ?= 8090
DATA_VOLUME ?= lowkey-data

.PHONY: docker-build docker-run docker-stop docker-logs docker-clean docker-rebuild \
       up up-minio down

## Build the Docker image
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

## Rebuild from scratch (no cache)
docker-rebuild:
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

## Run the container (detached)
docker-run:
	docker run -d \
		--name $(CONTAINER) \
		-p $(PORT):8090 \
		-v $(DATA_VOLUME):/data \
		$(IMAGE_NAME):$(IMAGE_TAG)

## Run interactively (foreground, auto-remove)
docker-run-it:
	docker run --rm -it \
		--name $(CONTAINER) \
		-p $(PORT):8090 \
		-v $(DATA_VOLUME):/data \
		$(IMAGE_NAME):$(IMAGE_TAG)

## Stop and remove the container
docker-stop:
	docker stop $(CONTAINER) && docker rm $(CONTAINER)

## Tail container logs
docker-logs:
	docker logs -f $(CONTAINER)

## Remove image and volume (destructive)
docker-clean:
	-docker rm -f $(CONTAINER) 2>/dev/null
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null
	@echo "Note: data volume '$(DATA_VOLUME)' preserved. Remove with: docker volume rm $(DATA_VOLUME)"

## Compose: build and start all services
up:
	docker compose up -d --build

## Compose: start with MinIO wired as S3 storage root
up-minio:
	docker compose -f docker-compose.yml -f docker-compose.minio.yml up -d --build

## Compose: stop all services
down:
	docker compose down

## Build the Go binary for Linux (without Docker)
build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o lowkeymediaserver .

## Build the Go binary for the current platform
build:
	go build -o lowkeymediaserver .
