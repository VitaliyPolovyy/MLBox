import asyncio
import base64
import io
import logging
import os
import sys

import ray
from PIL import Image
from ray import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.debug("testtt: debug")
logging.info("testtt: info")
logging.error("testtt: error")


# Initialize Ray and Ray Serve with log_to_driver=True
ray.init(log_to_driver=True)
serve.start(http_options={"host": "0.0.0.0", "port": 8000})


# Define the MLBox service with one replica
@serve.deployment(num_replicas=1, route_prefix="/mlbox")
class MLBox:

    async def process_image_1(self, image_data: bytes):
        try:
            # Save the image temporarily to check its validity
            with open("debug_image.png", "wb") as f:
                f.write(image_data)

            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify the image is not corrupted
            image = Image.open(io.BytesIO(image_data))  # Re-open for processing
            image = image.convert("L")  # Convert to grayscale

            # Proceed with further processing...
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            # Simulate creating JSON data for the image processing results
            json_data = {
                "table": [
                    {
                        "index": 1,
                        "category_id": 2,
                        "area": 3,
                        "accuracy": 4,
                        "major_diameter": 5,
                        "minor_diameter": 6,
                    }
                ],
                "status_code": "200",
                "message": "Image and data processed successfully",
                "image": image_base64,
            }

            # Send the image and processing results to the ERP system
            # await self.send_result_to_erp(image_data, json_data)

            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def send_result_to_erp(self, image_data: bytes, json_data: dict):
        """Send the image and processing results to the ERP system."""
        try:
            # Prepare the multipart form-data payload with the image and JSON data
            files = {
                "image": (
                    "processed_image.png",
                    image_data,
                    "image/png",
                )  # The image as a file in form-data
            }
            # Send POST request to the ERP API using multipart/form-data for the image and JSON for the payload
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://ite.roshen.com:4433/WS/api/_CV_TEST?call_in_async_mode=false",
                    # Replace with your ERP API endpoint
                    # files=files,  # Sending the image in form-data
                    json=json_data,  # JSON data to include in the payload
                    headers={
                        "accept": "application/xml"  # Adjust headers as needed for your ERP API
                    },
                )
                # Log the response status for debugging

        except Exception as e:
            print(f"Error sending result to ERP: {str(e)}")

    async def __call__(self, request):
        try:
            # Extract the image data and service code from the request
            form = await request.form()
            image_file = form["image"]
            image_data = await image_file.read()
            service_code = form["service_code"]

            # Immediately process the request without batching
            if service_code == "1":
                # Process the image immediately in the background
                asyncio.create_task(self.process_image_1(image_data))

                return {
                    "status": "success",
                    "message": "Image processing started in the background 2",
                }
            else:
                return {"status": "error", "message": "Unsupported service code"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Deploy the MLBox service
serve.run(MLBox.bind())
