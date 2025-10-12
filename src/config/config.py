from dotenv import dotenv_values
import os

env = dotenv_values(".env")
BASE_URL = os.environ.get('BASE_URL', 'http://localhost')
PORT = os.environ.get('PORT', 5000)
DEBUG = os.environ.get('DEBUG', True)
API_SNAILLY = os.environ.get('API_SNAILLY', 'https://snailly.id')
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/snailly-backend')