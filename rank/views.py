import re

from django.conf import settings
from django.contrib.auth import authenticate, login, logout, REDIRECT_FIELD_NAME
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.db.models import Sum
from django.shortcuts import render, redirect
from django.views import View
from django.views.generic import ListView, DetailView
from django.views.generic.detail import SingleObjectMixin

from gnnrec.kgrec import recall, rank
from .models import Author, Paper


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


# 召回和学者排名模块上下文对象，在RankConfig.ready()中初始化
recall_ctx = None
rank_ctx = None


class SearchPaperView(LoginRequiredMixin, ListView):
    template_name = 'rank/search_paper.html'

    def get_queryset(self):
        if not self.request.GET.get('q'):
            return Paper.objects.none()
        _, pid = recall.recall(recall_ctx, self.request.GET['q'], settings.PAGE_SIZE)
        return sorted(Paper.objects.filter(id__in=pid), key=lambda p: pid.index(p.id))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['q'] = self.request.GET.get('q', '')
        return context


class PaperDetailView(LoginRequiredMixin, DetailView):
    model = Paper


# 参考 https://docs.djangoproject.com/en/3.2/topics/class-based-views/mixins/#using-singleobjectmixin-with-listview
class AuthorDetailView(LoginRequiredMixin, SingleObjectMixin, ListView):
    template_name = 'rank/author_detail.html'
    paginate_by = settings.PAGE_SIZE

    def get(self, request, *args, **kwargs):
        self.object = self.get_object(queryset=Author.objects.all())
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        return self.object.papers.order_by('-n_citation')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['author'] = self.object
        context['n_citation'] = self.object.papers.aggregate(Sum('n_citation'))['n_citation__sum']
        return context


class SearchAuthorView(LoginRequiredMixin, ListView):
    template_name = 'rank/search_author.html'

    def get_queryset(self):
        if not self.request.GET.get('q'):
            return Author.objects.none()
        return Author.objects.filter(name=self.request.GET['q'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['q'] = self.request.GET.get('q', '')
        return context


class AuthorRankView(LoginRequiredMixin, ListView):
    template_name = 'rank/author_rank.html'

    def get_queryset(self):
        if not self.request.GET.get('q'):
            return Author.objects.none()
        _, aid = rank.rank(rank_ctx, self.request.GET['q'])
        return sorted(Author.objects.filter(id__in=aid), key=lambda a: aid.index(a.id))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['q'] = self.request.GET.get('q', '')
        return context
