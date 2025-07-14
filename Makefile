train:
	uv run src/training_pipeline/training.py 

training-data:
	uv run src/feature_pipeline/preprocessing/core.py 

frontend:
	uv run streamlit run src/inference_pipeline/frontend/main.py --server.port 8501


# Backfilling the Feature Store
backfill-features:
	uv run src/inference_pipeline/backend/backfill_feature_store.py --scenarios start end --target features 
	
backfill-predictions:
	uv run src/inference_pipeline/backend/backfill_feature_store.py --scenarios start end --target predictions

backfill-all: backfill-features backfill-predictions	
	

# Docker
start-docker:
	sudo systemctl start docker

image:
	docker build -t divvy-hourly .

container:
	docker run -it --env-file .env -p 8501:8501/tcp divvy-hourly:latest 


# Git
push-main:
	git push -u codeberg main 
	git push rad main 

