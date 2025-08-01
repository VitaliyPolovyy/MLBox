=== RAY ==================
ray start --head --dashboard-host=0.0.0.0
python deployments/peanut\_deployment.py tail -f /dev/nul

sudo ss -tulnp | grep 8000

Dashboard:
172.29.46.134:8265

=== PEANUTS REQUEST ==================



PS:

curl.exe -X POST -F "image=@C:\\temp\\D21010101000001\_\_14972\_\_V223-011\_\_0003.jpg" -F 'json={"service\_code": "1", "alias": "DMS", "key": "   9127673     1", "response\_method": "HTTP\_POST\_REQUEST", "response\_endpoint": "https://ite.roshen.com:4433/WS/api/\_MLBOX\_HANDLE\_RESPONSE?call\_in\_async\_mode=false"}'  http://10.11.192.143:8000/peanuts/process\_image



PS:

curl.exe -X POST -F "image=@C:\\temp\\D21010101000001\_\_14972\_\_V223-011\_\_0003.jpg" -F 'json={"service\_code": "1", "alias": "DMS", "key": "   9127673     1", "response\_method": "HTTP\_POST\_REQUEST", "response\_endpoint": "https://ite.roshen.com:4433/WS/api/\_MLBOX\_HANDLE\_RESPONSE?call\_in\_async\_mode=false"}'  http://172.29.46.134:8000/peanuts/process\_image





CMD:

C:\\Users\\vitaliy.polovoy>curl.exe http://172.29.46.134:8000/

{"status":"error","message":"Request parsing error: 'image'Request parsing error: 'image'"}

curl.exe -X POST -F "image=@C:\\temp\\D21010101000001\_\_14972\_\_V223-011\_\_0003.jpg" -F "json={\\"service\_code\\": \\"1\\", \\"alias\\": \\"DMS\\", \\"key\\": \\" 9127673 1\\", \\"response\_method\\": \\"HTTP\_POST\_REQUEST\\", \\"response\_endpoint\\": \\"https://ite.roshen.com:4433/WS/api/\_MLBOX\_HANDLE\_RESPONSE?call\_in\_async\_mode=false\\"}" http://172.29.46.134:8000/peanuts/process\_image




=== OTHER ==================
netsh interface portproxy show all


docker build -t mlbox:latest .

docker rm -f mlbox

docker run -d --name mlbox \
  -p 8001:8000 -p 8265:8265 \
  -e HF_TOKEN=$HF_TOKEN \
  -v /home/dev/MLBox/artifacts:/app/artifacts \
  -v /home/dev/MLBox/tmp:/app/tmp \
  -v /home/dev/MLBox/logs:/app/log \
  mlbox


docker logs -f mlbox





=== PROD-DEPLOY ==================
WSL + Docker

docker build -t mlbox:latest .
docker save -o "/mnt/c/My storage/Python projects/MLBox/assets/docker/mlbox.tar" mlbox
scp "/mnt/c/My storage/Python projects/MLBox/assets/docker/mlbox.tar" vitaliy@10.11.105.80:/home/vitaliy/docker_images


MLBox
sudo systemctl stop mlbox
docker rm mlbox
docker rmi mlbox
docker load -i mlbox.tar
sudo systemctl start mlbox

if docker ps --filter name=cleanhands --format "{{.Status}}" | grep -q "healthy" && curl -sf http://localhost:8000/status >/dev/null 2>&1; then echo "✅ CleanHands: RUNNING & HEALTHY"; else echo "❌ CleanHands: PROBLEM DETECTED"; fi
http://10.11.105.80:8000/client/

