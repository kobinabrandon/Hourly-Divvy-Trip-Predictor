# Preprocessing
training-data:
	uv run src/feature_pipeline/preprocessing/core.py 

# Model Training
train:
	uv run src/training_pipeline/training.py 

# Backfilling the Feature Store
backfill-features:
	uv run src/inference_pipeline/backend/backfill_feature_store.py --scenarios start end --target features 
	
backfill-predictions:
	uv run src/inference_pipeline/backend/backfill_feature_store.py --scenarios start end --target predictions
	
backfill-all: backfill-features backfill-predictions	


# Frontend
frontend:
	uv run streamlit run src/inference_pipeline/frontend/main.py --server.port 8501

start-docker:
	sudo systemctl start docker

image:
	docker build -t divvy-hourly .

container:
	docker run -it --env-file .env -p 8501:8501/tcp divvy-hourly:latest 

