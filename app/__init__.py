from flask import Flask
from flask_pymongo import PyMongo
#from app.model import Model

mongo = PyMongo()

def init_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('config.Config')
    mongo.init_app(app)
    #model = Model(app)

    with app.app_context():
        from app.api import cad_api, api_bp
        app.register_blueprint(api_bp)
    #print(app.url_map)
    return app