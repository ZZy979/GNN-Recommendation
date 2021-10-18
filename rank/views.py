import logging
import re

from django.conf import settings
from django.contrib.auth import authenticate, login, logout, REDIRECT_FIELD_NAME
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.views import View
from django.views.generic import ListView, DetailView

import gnnrec.kgrec.recall as recall
from .models import Paper

logger = logging.getLogger(__name__)


class LoginView(View):

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('rank:index')
        return render(request, 'rank/login.html', {'login_url': request.get_full_path()})

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(self.get_redirect_url())
        else:
            return render(request, 'rank/login.html', {'message': '用户名或密码错误'})

    def get_redirect_url(self):
        return self.request.POST.get(REDIRECT_FIELD_NAME) \
               or self.request.GET.get(REDIRECT_FIELD_NAME, 'rank:index')


def logout_view(request):
    logout(request)
    return redirect('rank:login')


class RegisterView(View):

    def get(self, request):
        return render(request, 'rank/register.html')

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = ''

        if not re.fullmatch('[0-9A-Za-z_]+', username):
            message = '用户名只能包含字母、数字和下划线'
        elif User.objects.filter(username=username).exists():
            message = '用户名已存在'
        elif password != password2:
            message = '两次密码不一致'

        if message:
            return render(request, 'rank/register.html', {'message': message})
        User.objects.create_user(username, email, password, first_name=name)
        return redirect('rank:login')


@login_required
def index(request):
    return render(request, 'rank/index.html')


recall_ctx = None


class SearchPaper(LoginRequiredMixin, ListView):
    template_name = 'rank/search_paper.html'

    def get_queryset(self):
        global recall_ctx
        if not self.request.GET.get('q'):
            self.queryset = Paper.objects.none()
        else:
            if recall_ctx is None:
                logger.info('正在加载模型和论文向量...')
                recall_ctx = recall.get_context(settings.PAPER_EMBEDS_FILE, settings.SCIBERT_MODEL_FILE)
            pid = recall.recall(recall_ctx, self.request.GET['q'], settings.PAGE_SIZE)[1].tolist()
            self.queryset = sorted(Paper.objects.filter(id__in=pid), key=lambda p: pid.index(p.id))
        return super().get_queryset()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['q'] = self.request.GET.get('q', '')
        return context


class PaperDetail(LoginRequiredMixin, DetailView):
    model = Paper
