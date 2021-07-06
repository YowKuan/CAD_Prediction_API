from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

class Config:
    """Set Flask configuration from .env file."""

    # General Config
    SECRET_KEY = environ.get('SECRET_KEY')
    FLASK_APP = environ.get('FLASK_APP')
    FLASK_ENV = environ.get('FLASK_ENV')

    #Database
    MONGO_URI = environ.get('MONGO_URI')
    # MONGODB= environ.get('MONGODB')
    # MONGODB_HOST= environ.get('MONGODB_HOST')
    # MONGODB_PORT = environ.get('MONGODB_PORT')
