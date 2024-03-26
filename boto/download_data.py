import os
import boto3
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()
    aws_access_key_id = os.getenv('aws_access_key_id')
    aws_secret_access_key = os.getenv('aws_secret_access_key')

    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        region_name='ru-msk',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url='https://hb.ru-msk.vkcs.cloud'
    )

    s3_client.download_file(
        'diploma-storage',
        'data/animals10',
        '../data/raw/animals10'
    )