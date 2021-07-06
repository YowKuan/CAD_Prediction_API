from flask import current_app as app
from flask import request, Blueprint, make_response
from flask.templating import render_template
from app import mongo
from . import Preprocess, Predict, Predict_techniques, mongoTypeHelper
import collections
from flask_restful import Resource, Api
import json
from bson import json_util



api_bp = Blueprint('api', __name__)
cad_api = Api(api_bp)

class SHAP_conclu(Resource):
    def get(self):
        patient_id = request.args.get('id')
        date = request.args.get('date')
        print(patient_id, date)
        string = date+patient_id
        return make_response(render_template('conclu{}.html'.format(string)), 200 )

class SHAP_techniques(Resource):
    def get(self):
        patient_id = request.args.get('id')
        date = request.args.get('date')
        print(patient_id, date)
        string = date+patient_id
        return make_response(render_template('technical{}.html'.format(string)), 200)





class CAD_analysis(Resource):
    def __init__(self):
        self.patient_collection = mongo.db.Patients
    def get(self):
        id = request.args.get('id') 
        date = request.args.get('date')
        data = self.patient_collection.find_one({"patient_id": id, "date":date})
        return json.loads(json_util.dumps(data))
    def post(self):
        
        data = request.get_json()
        #print(data)
        Preprocess.convert_json_to_csv(data)
        pre_processed = Preprocess.preprocess()
        result_cli_conclu = Predict.predict()
        result_techniques = Predict_techniques.predict()
        shap_conclu, date_conclu = Predict.Shap_text()
        shap_techniques, date_techniques = Predict_techniques.Shap_text()

        all_data = []
        
        patients = collections.defaultdict(list)
        for cli_key, tech_key in zip(result_cli_conclu.keys(), result_techniques.keys()):
            results = {"patient_id":cli_key, 
                    "date":date_conclu[cli_key],
                    "cil_conclu":result_cli_conclu[cli_key], 
                    "techniques":result_techniques[tech_key],
                    "overall": (result_cli_conclu[cli_key]*3 + result_techniques[tech_key]*2)/5,
                    "shap_cli_conclu": shap_conclu[cli_key],
                    "shap_techniques": shap_techniques[tech_key]
                    }
            results = mongoTypeHelper.correct_encoding(results)
            all_data.append(results)
        for data in all_data:
            self.patient_collection.insert_one(data)

        return str(all_data[0])

cad_api.add_resource(CAD_analysis, '/cad')
cad_api.add_resource(SHAP_conclu, '/shap/conclu')
cad_api.add_resource(SHAP_techniques, '/shap/techniques')




# @app.route('/input', methods=['POST'])
# def input_data():
#     return "test"
    # patient_collection = mongo.db.Patients
    # data = request.get_json()
    # #print(data)
    # Preprocess.convert_json_to_csv(data)
    # pre_processed = Preprocess.preprocess()
    # result_cli_conclu = Predict.predict()
    # result_techniques = Predict_techniques.predict()
    # shap_conclu, date_conclu = Predict.Shap_text()
    # shap_techniques, date_techniques = Predict_techniques.Shap_text()

    # all_data = []
    
    # patients = collections.defaultdict(list)
    # for cli_key, tech_key in zip(result_cli_conclu.keys(), result_techniques.keys()):
    #     results = {"patient_id":cli_key, 
    #             "date":date_conclu[cli_key],
    #             "cil_conclu":result_cli_conclu[cli_key], 
    #             "techniques":result_techniques[tech_key],
    #             "overall": (result_cli_conclu[cli_key]*3 + result_techniques[tech_key]*2)/5,
    #             "shap_cli_conclu": shap_conclu[cli_key],
    #             "shap_techniques": shap_techniques[tech_key]
    #             }
    #     results = mongoTypeHelper.correct_encoding(results)
    #     all_data.append(results)

    # return str(all_data[0])

# @app.route('/shapconclu', methods=['GET'])
# def shapconclu():
#     patient_id = request.args.get('id')
#     date = request.args.get('date')
#     print(patient_id, date)
#     string = date+patient_id
#     return render_template('conclu{}.html'.format(string))

# @app.route('/shaptechniques', methods=['GET'])
# def shaptechniques():
#     patient_id = request.args.get('id')
#     date = request.args.get('date')
#     print(patient_id, date)
#     string = date+patient_id
#     return render_template('technical{}.html'.format(string))
