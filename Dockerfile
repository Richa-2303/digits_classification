FROM python:3.9.17
EXPOSE 5000
WORKDIR /digit_classification
COPY . /digit_classification/
RUN pip3 install -r /digit_classification/requirements.txt
RUN pytest -c descision_trees
RUN python -m exp "[svm,descision_trees,logistic_regression]" 0.2 0.2 1
# VOLUME /digit_classification/models
ENV FLASK_APP=hello.py
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]