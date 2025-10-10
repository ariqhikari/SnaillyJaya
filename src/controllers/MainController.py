
from flask import Blueprint
from flask import request,g
from src.services.CleanDataServices import CleanDataService
from src.services.CreateScreenshoot import ScreenshotDataService
from src.services.PredictDataServices import PredictDataService
from src.services.HITLServices import HITLService
import src.utils.getResponse as Response  

MainApp = Blueprint('MainApp', __name__,)
cleanDataService =  CleanDataService()
screenshotService = ScreenshotDataService()
predictDataService = PredictDataService()
hitlService = HITLService()
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
    data = request.json

    token = data.get('token')
    parent_id = data.get('parent_id')
    child_id = data.get('child_id')
    image_folder = data.get('image_folder')

    if not image_folder:
        return Response.error("image_folder wajib diisi", 400)
    if not token:
        return Response.error("token wajib diisi", 400)

    result = screenshotService.createScreenshotData(image_folder, {
        "parent_id": parent_id,
        "child_id": child_id,
        "token": token
    })
    print(result)
    # return Response.error("token wajib diisi", 400)

    if result['status'] == 'failed':
        return Response.error(result['data'], result['code'])
    return Response.success(result['data'], "success screenshoot")
               
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
