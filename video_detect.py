def detect_who(model, face_image):
    # 예측
    name = ""
    result = model.predict(face_image)
    # result_msg = f"시베리안 허스키 가능성 : {result[0][0]*100: .3f}% / 골든리트리버 가능성 : {result[0][1]*100: .3f}% / 진돗개 가능성 : {result[0][2]*100: .3f}% / 푸들 가능성 : {result[0][3]*100: .3f}% / 불독 가능성 : {result[0][4]*100: .3f}% / 시츄 가능성 : {result[0][5]*100: .3f}% / 치와와 가능성 : {result[0][6]*100: .3f}% / 비글 가능성 : {result[0][7]*100: .3f}%"
    # result_msg = [result[0][0]*100,result[0][1]*100,result[0][2]*100,result[0][3]*100,result[0][4]*100,result[0][5]*100,result[0][6]*100,result[0][7]*100]
    # result_msg = [round(result[0][0]*100,2),round(result[0][1]*100,2),round(result[0][2]*100,2),round(result[0][3]*100,2),round(result[0][4]*100,2),round(result[0][5]*100,2),round(result[0][6]*100,2),round(result[0][7]*100,2)]
    result_msg = [["시베리안 허스키",round(result[0][0]*100,2)],["골든리트리버",round(result[0][1]*100,2)],["진돗개",round(result[0][2]*100,2)],["푸들",round(result[0][3]*100,2)],["불독",round(result[0][4]*100,2)],["시츄",round(result[0][5]*100,2)],["치와와",round(result[0][6]*100,2)],["비글",round(result[0][7]*100,2)]]
    
    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Siberian"
    elif name_number_label == 1:
        name = "Golden Retriver"
    elif name_number_label == 2:
        name = "Jindo"
    elif name_number_label == 3:
        name = "Foodle"
    elif name_number_label == 4:
        name = "Bulldog"
    elif name_number_label == 5:
        name = "sichu"
    elif name_number_label == 6:
        name = "Chiwawa"
    elif name_number_label == 7:
        name = "Biggle"
    return (name,result_msg)

def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img =''
   
    cascade_file_path = settings.CASCADE_FILE_PATH
    model_file_path = settings.MODEL_FILE_PATH
    
    model = keras.models.load_model(model_file_path)
    image = np.asarray(Image.open(upload_image))

    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


    # CascadeClassifier 생성
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCV 활용 어굴인식 함수 호출 detectMultiScale()
    faces = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=5, minSize=(64,64))
    # 얼굴이 1개 이상 검출된 경우
    
    if len(faces) > 0:
        count = 1
        # print(f"인식된 얼굴의 수 : {len(faces)}")
        for (x_pos, y_pos, width, height) in faces: # x_pos, y_pos, width, height
            face_image = image_rgb[y_pos: y_pos+height, x_pos:x_pos+width]
            print(f"인식한 얼굴의 사이즈 : {face_image.shape}")
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                print("인식한 얼굴의 사이즈가 너무 작습니다.")
                continue
            else:
                # 인식한 얼굴 사이즈 축소
                face_image = cv2.resize(face_image,(64,64))

            # 인식한 얼굴 주변에 붉은색 사각형을 표시
            cv2.rectangle(image_rgb, (x_pos, y_pos),(x_pos+width, y_pos+height), (255,0,0), thickness=2)
             # image, (사각형 시작 좌표), (사각형 종료 좌표), (색상), (선 굵기)thickness=

            # 인식한 얼굴을 1장의 사진으로 합치고 --> 배열로 변환
            face_image = np.expand_dims(face_image, axis=0)
            # 인식한 얼굴에 이름을 표기
            name, result_list = detect_who(model, face_image)
            cv2.putText(image_rgb, name, (x_pos,y_pos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),2)
            # result_list.append(result)
            count = count + 1 
        is_success, img_buffer = cv2.imencode(".png", image_rgb)
        if is_success:
            # image -> binary 형대임
            io_buffer = io.BytesIO(img_buffer)
            result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'","")
    else:
        print("이미지 파일에 얼굴이 없습니다.")

    backend.clear_session()# tensorflow session 종료
    
    return(result_list, result_name, result_img)
    
cap = cv2.VideoCapture(0)

while True:
    #카메라로 부터 사진 한장 읽기 
    ret, frame = cap.read()
    # 얼굴 검출 시도 
    image, face = face_detector(frame)
    try:
        #검출된 사진을 흑백으로 변환 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. 
        if result[1] < 500:
            #????? 어쨋든 0~100표시하려고 한듯 
            confidence = int(100*(1-(result[1])/300))
            # 유사도 화면에 표시 
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked! 
        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
           #75 이하면 타인.. Locked!!! 
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        #얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()