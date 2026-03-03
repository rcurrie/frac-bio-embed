train-local-docker:
	docker run --rm \
	-v ~/.aws:/root/.aws:ro \
	-e S3_BASE=s3://braingeneers/personal/$$USER/frac-bio-embed \
	-e AWS_ENDPOINT_URL=http://host.docker.internal:30000 \
	frac-bio-embed \
	--max-samples 1000 --epochs 2

run-local-s3-job:
	kubectl config use-context docker-desktop
	envsubst < job.yaml | kubectl apply -f -
	kubectl wait --for=condition=complete --timeout=120s job/frac-bio-embed-train
	kubectl logs job/frac-bio-embed-train
	kubectl delete job/frac-bio-embed-train