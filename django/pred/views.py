from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect
from django.http import HttpResponse
from .models import Member
from django.utils import timezone
from .forms import JoinForm,LoginForm

class PredView(TemplateView):
    # 생성자
    def __init__(self):
        self.params = {'result_list':[],
        'result_name':"",
        'result_img':"",
        'form': ImageForm()}
    #GET requests (index.html 파일 초기 표시)
    def get(self, req):
        self.params['dis']= ['none','none','none','none','none','none','none','none']
        return render(req, 'pred/index2.html', self.params)
        
    #POST requests (index.html 파일에 결과 표시)
    def post(self, req):
        # POST method에 의해 전달되는 Form Data
        form = ImageForm(req.POST, req.FILES)
        # Form Data error check
        if not form.is_valid():
            raise ValueForm('invalid form')
        # Form Data에서 이미지 파일 얻기
        image = form.cleaned_data['image']
        # Image file을 지정해서 얼굴 인식
        result = detect(image)

        # 분류된 얼굴 결과 저장
        self.params['result_list'], self.params['result_name'],self.params['result_img'] = result
        self.params['dis']= ['none','none','none','none','none','none','none','none']
        li=[]
        for i in range(8):
            li.append(self.params['result_list'][i][1])
        idx=li.index(max(li))
        self.params['dis'][idx]='display'
        # 페이지에 화면 표시
        self.params['percent']=li

        return render(req, 'pred/index2.html', self.params)





def join(request):
    if request.method == 'GET':
        form = JoinForm()
        return render(request, 'member/join.html', {})
    else:
        form = JoinForm(request.POST)
        if form.is_valid():
            member = form.save(commit=False)
            member.c_date = timezone.now()
            member.save()
            return HttpResponse('가입 완료')
        else:
            return HttpResponse('오류')

def login(request):
    if request.method == 'GET':
        form = LoginForm()
        return render(request, 'member/login.html', {'form': form})
    else:
        user_id = request.POST['user_id']
        user_pw = request.POST['user_pw']
        try:
            Member.objects.get(user_id=user_id, user_pw=user_pw)
        except Member.DoesNotExist:
            return HttpResponse('로그인 실패')
        else:
            return HttpResponse('로그인 성공')


