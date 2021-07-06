from app import db
import datetime

class Patients(db.Documents):
    patient_id = StringField(required=True)
    date_modified = DateTimeField(default=datetime.datetime.utcnow)