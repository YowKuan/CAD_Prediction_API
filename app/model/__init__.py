from app.model.Patients import Patients

class Model:
    def __init__(self, app=None):
        if app is not None:
            self.init_app(app)
    def init_app(self, app):
        app.model = self

        self.Patients = Patients