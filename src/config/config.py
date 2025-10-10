from dotenv import dotenv_values

env = dotenv_values(".env")
BASE_URL = env["BASE_URL"] or "http://localhost"
PORT = env["PORT"] or 5000
DEBUG = env["DEBUG"] or True
API_SNAILLY = env["API_SNAILLY"] or "https://snailly.id"
DATABASE_URL = env["DATABASE_URL"] or "postgresql://postgres:postgres@localhost:5433/snailly-backend"
