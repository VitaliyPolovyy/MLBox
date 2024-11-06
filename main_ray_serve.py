import ray  # Add this import
from ray import serve
from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
import logging

ray.init()
serve.start()
@serve.deployment  # You missed this decorator
class MyFirstDeployment:
    def __init__(self, msg):  # Fix asterisks
        self.msg = msg

    def __call__(self, request: Request):  # Fix asterisks and use imported Request
        name = request.query_params.get("name", "stranger")
        logging.info(f"logging.info: Received request for name: {name}")
        print(f"print.info: Received request for name: {name}")
        return f"Hello {name}!"
    
MLBoxDeployment
ImageProcessing


my_first_deployment = MyFirstDeployment.bind("Hello world!")
handle = serve.run(my_first_deployment)

# This assertion will fail because __call__ now expects a request parameter
# and returns "Hello stranger!" instead of the msg
# You should test with curl instead:
# curl "http://localhost:8000/?name=Alice"