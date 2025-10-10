from flask import Flask, g
from flask_cors import CORS
from flask_migrate import Migrate
from src.config.config import BASE_URL, PORT, DEBUG
from src.config.database import database

main_app = Flask(
    __name__,
    static_folder="../../public",
    static_url_path="/",
)
CORS(main_app)

db = database(main_app)
migrate = Migrate(main_app, db)

@main_app.route('/')
def index():
    return "Welcome to the Snailly API"

print("Server is running on url: " + BASE_URL + ":", PORT)
if __name__ == "__main__":
    main_app.run(debug=DEBUG, host=BASE_URL, port=PORT)