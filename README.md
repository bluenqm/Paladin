# Paladin

## Online Demo
Online demo version here: http://130.88.110.27:8000/projects/

Access using minh/nqminhtest

## System Requirements
Paladin needs to use nVidia GPU for training models. You need to install CUDA 10.1.


## Installation using the source code

1. Get source code

2. Create a new virtual environment
```
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
```
3. Install requires package
```
pip3 install -r requirements.txt
```
4. Install apex
```
git clone https://www.github.com/nvidia/apex
cd apex
python setup.py install
```
5. Create database
```
python3 manage.py makemigrations server
python3 manage.py migrate
```
6. Create admin user
```
python3 manage.py createsuperuser
```
7. Run the application
```
python3 manage.py runserver
```
8. Use a web browser and navigate to 127.0.0.1:8000
