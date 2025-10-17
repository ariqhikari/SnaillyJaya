
from flask import Blueprint
from flask import request, g, jsonify
from src.services.CleanDataServices import CleanDataService
from src.services.CreateScreenshoot import ScreenshotDataService
from src.services.PredictDataServices import PredictDataService
from src.services.HITLServices import HITLService
import src.utils.getResponse as Response  
import os
import re
from groq import Groq

MainApp = Blueprint('MainApp', __name__,)
cleanDataService =  CleanDataService()
screenshotService = ScreenshotDataService()
predictDataService = PredictDataService()
hitlService = HITLService()

client = Groq(api_key="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")

@MainApp.route('/test', methods=['POST'])
def test():
    return Response.success([],"Test post endpoint is working")


@MainApp.route('/predict',methods=['POST'])
def predict_data():
    data = request.json
    result = predictDataService.createPredictData(data)
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success predict data")

@MainApp.route('/screenshoot', methods=['POST'])
def screenshoot():
    token = request.form.get('token')
    parent_id = request.form.get('parent_id')
    child_id = request.form.get('child_id')
    image_file = request.files.get('image_file')

    if not image_file:
        return Response.error("image_file wajib diisi", 400)
    if not token:
        return Response.error("token wajib diisi", 400)

    # Simpan file ke folder, misal 'public/screenshots'
    save_path = os.path.join('public', 'screenshots', image_file.filename)
    image_file.save(save_path)

    result = screenshotService.createScreenshotFromFile(save_path, {
        "parent_id": parent_id,
        "child_id": child_id,
        "token": token
    })
    print(result)

    if result['status'] == 'failed':
        return Response.error(result['data'], result['code'])
    return Response.success(result['data'], "success screenshoot")
               
@MainApp.route('/scrapping', methods=['POST'])
def clean_data():
    data = request.json
    url = data.get('url')
    parent_id = data.get('parent_id')
    child_id = data.get('child_id')
    result = cleanDataService.createCleanData({
        "url": url,
        "parent_id": parent_id,
        "child_id": child_id,
    })
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success predict url")

@MainApp.route('/retrain', methods=['POST'])
def retrain_model():
    result = predictDataService.createRetrainModel()
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success retrain model")

@MainApp.route('/update-label', methods=['PUT'])
def update_label():
    data = request.json
    id = data.get('id')
    new_label = data.get('new_label')

    result = hitlService.updatePredictLabelById(int(id), new_label)
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success update label")

@MainApp.route('/update-label-logid', methods=['PUT'])
def update_label_logid():
    data = request.json
    log_id = data.get('log_id')
    new_label = data.get('new_label')

    result = hitlService.updatePredictLabelByLogId(log_id, new_label)
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success update label")

@MainApp.route('/seed-dataset', methods=['POST'])
def seed_dataset():
    result = hitlService.createSeedDataset()
    if(result['status'] == 'failed'):
        return Response.error(result['data'],result['code'])
    return Response.success(result['data'],"success create seed dataset")

@MainApp.route('/predict-image', methods=['POST'])
def predict_image():
    print("INI MASUK SINI!!!!!!!!!!!!!")
    try:
        data = request.json
        image_url = data.get('image_url')

        print("Ini IMAGE URL nya: ", image_url)
        
        global client
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is this image related to porn or adult content? If it is porn or adult content, answer with 'porn' only; if it is not porn, answer with 'np'"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=3,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Mengubah teks menjadi lowercase
        text = completion.choices[0].message.content.lower()
        # Menghapus tanda baca menggunakan regex
        text = re.sub(r'[^\w\s]', '', text)
        return jsonify({"hasil": text})
    except Exception as e:
        print("INI ERROR : ", e)
        return jsonify({"error": str(e)}), 500