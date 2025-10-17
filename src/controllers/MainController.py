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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

#Tambahan untuk melakukan proses hasil screenshoot

@MainApp.route('/screenshoot', methods=['POST'])
def screenshoot():
    try:
        image_file = request.files.get('image_file')
        if not image_file:
            return Response.error("image_file wajib diisi", 400)

        # Convert file ke base64
        import base64
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # Format data:image/jpeg;base64,...
        image_url = f"data:image/jpeg;base64,{image_base64}"

        global client
        completion = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a protective parent who wants to keep children safe from harmful online content."
                                "Your task is to analyze the image and decide if it contains any of the following: "
                                "pornography, nudity, kissing, sexual acts, LGBT romantic or sexual content, violence, or gambling. "
                                "If the image includes any of these, answer only with 'berbahaya'. "
                                "If the image does not include any of these, answer only with 'aman'."
                            )#"Is this image related to porn or adult content? If it is porn or adult content, answer with 'porn' only; if it is not porn, answer with 'np'"
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
        import re
        text = completion.choices[0].message.content.lower()
        # Menghapus tanda baca menggunakan regex
        text = re.sub(r'[^\w\s]', '', text)
        # return jsonify({"hasil": text})
        return Response.success({"label": text}, "success predict image")
    except Exception as e:
        print("INI ERROR : ", e)
        return Response.error({"error": str(e)}, 500)


# @MainApp.route('/screenshoot', methods=['POST'])
# def screenshoot():
#     data = request.json
# @MainApp.route('/screenshoot', methods=['POST'])
# def screenshoot():
#     token = request.form.get('token')
#     parent_id = request.form.get('parent_id')
#     child_id = request.form.get('child_id')
#     image_file = request.files.get('image_file')

#     if not image_file:
#         return Response.error("image_file wajib diisi", 400)
#     if not token:
#         return Response.error("token wajib diisi", 400)

#     # Simpan file ke folder, misal 'public/screenshots'
#     save_path = os.path.join('public', 'screenshots', image_file.filename)
#     image_file.save(save_path)

#     result = screenshotService.createScreenshotFromFile(save_path, {
#         "parent_id": parent_id,
#         "child_id": child_id,
#         "token": token
#     })
#     print(result)

#     if result['status'] == 'failed':
#         return Response.error(result['data'], result['code'])
#     return Response.success(result['data'], "success screenshoot")
#     token = data.get('token')
#     parent_id = data.get('parent_id')
#     child_id = data.get('child_id')
#     image_path = data.get('image_path')

#     if not image_path:
#         return Response.error("image_path wajib diisi", 400)
#     if not token:
#         return Response.error("token wajib diisi", 400)

#     result = screenshotService.createScreenshotFromFile(image_path, {
#         "parent_id": parent_id,
#         "child_id": child_id,
#         "token": token
#     })
#     print(result)

#     if result['status'] == 'failed':
#         return Response.error(result['data'], result['code'])
#     return Response.success(result['data'], "success screenshoot")
               
@MainApp.route('/scrapping', methods=['POST'])
def clean_data():
    data = request.json
    url = data.get('url')
    parent_id = data.get('parent_id')
    child_id = data.get('child_id')
    token = data.get('token')
    result = cleanDataService.createCleanData({
        "url": url,
        "parent_id": parent_id,
        "child_id": child_id,
        "token": token
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
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                             "text": (
                                "You are a protective parent who wants to keep children safe from harmful online content."
                                "Your task is to analyze the image and decide if it contains any of the following: "
                                "pornography, nudity, kissing, sexual acts, LGBT romantic or sexual content, violence, or gambling. "
                                "If the image includes any of these, answer only with 'berbahaya'. "
                                "If the image does not include any of these, answer only with 'aman'."
                            )#"Is this image related to porn or adult content? If it is porn or adult content, answer with 'porn' only; if it is not porn, answer with 'np'"
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
        return Response.success({"label": text}, "success predict image")
    except Exception as e:
        print("INI ERROR : ", e)
        return Response.error({"error": str(e)}, 500)
