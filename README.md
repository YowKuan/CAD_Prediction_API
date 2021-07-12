
# CAD API

This API is designed to analyze Coronary Heart Disease patients' risk of mortality
based on clinical notes. It also provides SHAP text plot to provide explainability.





## Installation

- Install CAD API with pip or conda (recommended).

```bash
# using pip
python3 -m venv <env>
source <env>/bin/activate
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```

- Modify SHAP module to enable output of HTML file.

```bash
cd C:/Users/<your user name>/anaconda3/envs/<env_name>/Lib/site-packages/shap/plots
```
```python
def text(shap_values, num_starting_labels=0, group_threshold=1, separator='', xmin=None, xmax=None, cmax=None):
    return out #add this line at the bottom of text function
    #display(HTML(out)) #comment this line
```

- Setup MongoDB Atlas database at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- Create .env file: 
```bash
FLASK_APP = wsgi.py
FLASK_ENV= development
SECRET_KEY = <your_secret_key>
MONGO_URI = <your_MongoDB_URI>
```


    
## Activate Server

```bash
python wsgi.py
```

  
## API Reference

#### Get analyze result based on patient_id and date

```http
  GET /cad/${id}${date}
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `id` | `string` | **Required**. patient_id |
| `date` | `string` | **Required**. operation date |

```http
  POST /cad/
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `身份證號` | `string` | **Required**. patient's id |
| `性別` | `string` | **Required**. patient's gender|
| `AGE` | `int` | **Required**. patient's age |
| `DATE` | `string` | **Required**. operation date |
| `Indication` | `string` | **Required**. |
| `Tech` | `string` | **Required**. Techniques |
| `Conclusion` | `string` | **Required**. operation date |


  