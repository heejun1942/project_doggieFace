from django.forms import forms, TextInput, PasswordInput
from django import forms
from .models import Member

class ImageForm(forms.Form):
    image = forms.ImageField(label="",
    error_messages={'missing' : '이미지 파일이 선택되지 않았습니다.',
    'invalid' : '분류할 이미지 파일을 선택해 주세요.',
    'invalid_image' : '이미지 파일이 아닙니다.'})

class JoinForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = ['user_id', 'user_pw', 'user_name']
        widgets = {
            'user_id': TextInput(),
            'user_pw': PasswordInput(),
            'user_name': TextInput()
    }

class LoginForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = ['user_id', 'user_pw']
        widgets = {
            'user_id': TextInput(),
            'user_pw': PasswordInput(),
    }