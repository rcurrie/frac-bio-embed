train-local:
	uv run python train.py --max-samples 100000 --epochs 20

docker-build:
	docker build \
		-t gitlab-registry.nrp-nautilus.io/rcurrie/frac-bio-embed \
		--platform linux/amd64 \
		-f Dockerfile .

docker-push:
	docker push gitlab-registry.nrp-nautilus.io/rcurrie/frac-bio-embed

train-local-docker:
	docker run --rm \
	-v ~/.aws:/root/.aws:ro \
	-e S3_BASE=s3://braingeneers/personal/$$USER/frac-bio-embed \
	-e AWS_ENDPOINT_URL=http://host.docker.internal:30000 \
	gitlab-registry.nrp-nautilus.io/rcurrie/frac-bio-embed \
	--max-samples 1000 --epochs 2

run-local-s3-job:
	kubectl config use-context docker-desktop
	-kubectl delete job/frac-bio-embed-train 2>/dev/null
	envsubst < job.yaml | kubectl apply -f -
	kubectl wait --for=condition=ready --timeout=-1s pod -l job-name=frac-bio-embed-train
	kubectl logs -f job/frac-bio-embed-train
# 	kubectl delete job/frac-bio-embed-train

run-cloud-s3-job:
	kubectl config use-context nautilus
	-kubectl delete job/frac-bio-embed-train 2>/dev/null
	envsubst < job.yaml | kubectl apply -f -
	kubectl wait --for=condition=ready --timeout=-1s pod -l job-name=frac-bio-embed-train

shell-into-cloud:
	kubectl exec -it frac-bio-embed-train-p4l8p -- /bin/bash 

tail-cloud-s3-job:
	kubectl logs -f job/frac-bio-embed-train

cleanup-cloud-s3-job:
	kubectl delete job/frac-bio-embed-train

list-s3-outputs:
	aws s3 ls --profile=braingeneers \
		s3://braingeneers/personal/rcurrie/frac-bio-embed/