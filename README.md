# Paladin
1. Get source code

2. Create a new virtual environment

python3 -m venv /path/to/venv

source /path/to/venv/bin/activate

3. Install requires package

pip3 install -r requirements.txt

4. Install apex

git clone https://www.github.com/nvidia/apex

cd apex

python setup.py install

5. Create database

python3 manage.py makemigrations server

python3 manage.py migrate

6. Create admin user

python3 manage.py createsuperuser
