sudo docker build -t exp:v1 -f docker/Dockerfile .
sudo docker run -v /mnt/d/Projects/ML-OPS/digit_classification/models:/digit_classification/models -it exp:v1 bash