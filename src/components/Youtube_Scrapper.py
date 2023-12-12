from src.logger import logging
from src.exception import CustomException
import os
import sys
from dotenv import load_dotenv
import googleapiclient.discovery
import pandas as pd

load_dotenv()

class Get_Youtube_comments:
    def __init__(self):
        pass

    def IngestData(self, Id):
        logging.info("Initiate the extraction of tweets from the X API")

        try:
            service_name = "youtube"
            api_version = "v3"
            DEVELOPER_KEY = os.getenv("API_KEY")

            youtube = googleapiclient.discovery.build(
                service_name, api_version, developerKey=DEVELOPER_KEY
            )

            request = youtube.commentThreads().list(
            part="id,snippet,replies",
            videoId=Id,
            maxResults=500,
        )
            response = request.execute()

            comments = []

            while response:
                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append([
                        comment["authorDisplayName"],
                        comment["publishedAt"],
                        comment["updatedAt"],
                        comment["likeCount"],
                        comment["textDisplay"]
                    ])
                
                if 'nextPageToken' in response:
                    request = youtube.commentThreads().list(
                    part="id,snippet,replies",
                    videoId=Id,
                    maxResults=500,
                    pageToken=response["nextPageToken"]
                    )
                    response = request.execute()
                else:
                    break

            df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

            return df

        except Exception as e:
            raise CustomException(e, sys)