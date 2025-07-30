=== RAY ==================
ray start --head --dashboard-host=0.0.0.0
serve deploy deployments/ray_serv.yaml
.venv/bin/python deployments/peanut_deployment.py

sudo ss -tulnp | grep 8000

Dashboard: 
172.29.46.134:8265

=== PEANUTS REQUEST ==================
curl.exe -X POST -F "image=@C:\temp\D21010101000001__14972__V223-011__0003.jpg" -F 'json={\"service_code\": \"1\", \"alias\": \"DMS\", \"key\": \"   9127673     1\", \"response_method\": \"HTTP_POST_REQUEST\", \"response_endpoint\": \"https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false\"}'  http://10.11.192.143:8000/peanuts/process_image

curl.exe -X POST -F "image=@C:\temp\D21010101000001__14972__V223-011__0003.jpg" -F 'json={\"service_code\": \"1\", \"alias\": \"DMS\", \"key\": \"   9127673     1\", \"response_method\": \"HTTP_POST_REQUEST\", \"response_endpoint\": \"https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false\"}'  http://172.29.46.134:8000/peanuts/process_image

=== OTHER ==================
netsh interface portproxy show all
