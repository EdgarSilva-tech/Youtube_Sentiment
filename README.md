YouTube Comment Sentiment Analysis
This project is a web application that performs sentiment analysis on YouTube video comments. It uses FastAPI for the backend, Docker for containerization, and a machine learning model for sentiment prediction.

Features: 
- Fetch comments from a YouTube vídeo 
- Perform sentiment analysis on the comments 
- Display results in a web interface 

Project Structure: 
- `compose.yaml`: Docker Compose configuration file 
- `Dockerfile`: Docker configuration for the application 
- `main.py`: FastAPI application entry point 
- `setup.py`: Python package setup file 
- `exception.py`: Custom exception handling 
- `logger.py`: Logging configuration 
- `utils.py`: Utility functions 
- `data_ingestion.py`: Data ingestion pipeline 
- `predict.py`: Sentiment prediction logic 
- `training.py`: Model training pipeline 
- `Youtube_Scrapper.py`: YouTube comment scraping functionality  

Setup and Installation: 
Clone the repository: git clone https://github.com/EdgarSilva-tech/Youtube_Sentiment.git. Install Docker. Create a `.env` file and add your YouTube API key: API_KEY=your_youtube_api_key_here. Build and run the Docker container: docker-compose up --build.

Usage: Access the web interface at `http://localhost:8000`. Enter a YouTube video ID in the input field. Submit to fetch comments and perform sentiment analysis. View the results displayed on the page. Model Training - The sentiment analysis model is trained using the `sentiment140` dataset. To retrain the model: Run the data ingestion script:  python -m src/components/data_ingestion, this will automatically trigger the model training process.
