import os

import boto3

s3_url = os.environ["S3_URL"]
s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://{s3_url}:9000",
    aws_access_key_id=os.environ["S3_KEY_ID"],
    aws_secret_access_key=os.environ["S3_ACCESS_KEY"],
    region_name="us-east-1",
)


# def list_files(bucket_name: str, prefix: str) -> List[str]:
#     paginator = s3_client.get_paginator("list_objects_v2")
#     page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
#     files = []
#     for page in page_iterator:
#         print(page)
#         if "Contents" in page:
#             for obj in page["Contents"]:
#                 key = obj["Key"]
#                 files.append(key)
#     return files


# def read_text_file(bucket_name: str, key: str) -> str:
#     response = s3_client.get_object(Bucket=bucket_name, Key=key)
#     content = response["Body"].read().decode("utf-8")
#     return content


# def save_pdf_file(bucket_name: str, key: str, destination_folder: str) -> None:
#     # Get the file name from the S3 key
#     file_name = os.path.basename(key)
#     file_path = os.path.join(destination_folder, file_name)
#
#     # Download the file from S3 and save it locally
#     response = s3_client.get_object(Bucket=bucket_name, Key=key)
#
#     # Read the content of the file
#     pdf_content = response["Body"].read()
#
#     # Save the content to the local file
#     with open(file_path, "wb") as pdf_file:
#         pdf_file.write(pdf_content)
#
#     # print(f"Saved {file_name} to {file_path}")


def download_bucket(name):
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=name, Prefix="")
    files = []
    for page in page_iterator:
        print(page)
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                files.append(key)

                response = s3_client.get_object(Bucket=name, Key=key)
                pdf_content = response["Body"].read()

                if not os.path.exists("bucket_content"):
                    os.makedirs("bucket_content")

                file_path = os.path.join("bucket_content", os.path.basename(key))
                with open(file_path, "wb") as pdf_file:
                    pdf_file.write(pdf_content)

    return files
