worker: celery -A sky_finance.scheduler.celery_app worker --loglevel=info --queues=ingestion,pipeline,storage,strategies,notifications,default --concurrency=4
beat: celery -A sky_finance.scheduler.celery_app beat --loglevel=info
flower: celery -A sky_finance.scheduler.celery_app flower --port=5555
web: uvicorn sky_finance.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
