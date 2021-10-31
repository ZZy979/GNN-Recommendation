from unittest.mock import patch
from urllib.parse import quote

from django.conf import settings
from django.contrib.auth import SESSION_KEY
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .models import Author, Paper


def create_test_data():
    User.objects.create_user('alice', 'alice@example.com', '1234')
    Author.objects.bulk_create([Author(id=i, name=f'A{i}') for i in range(3)])
    papers = Paper.objects.bulk_create([
        Paper(id=i, title=f'P{i}', year=2021, abstract='', n_citation=2 - i)
        for i in range(3)
    ])
    for i, a in enumerate([[0], [0, 1], [1, 2]]):
        papers[i].authors.set(a)
    for i, r in enumerate([[], [0], [0, 1]]):
        papers[i].references.set(r)


class LoginViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def test_get(self):
        response = self.client.get(reverse('rank:login'))
        self.assertTemplateUsed(response, 'rank/login.html')

    def test_get_already_login(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})
        response = self.client.get(reverse('rank:login'))
        self.assertRedirects(response, reverse('rank:index'))

    def test_ok(self):
        data = {'username': 'alice', 'password': '1234'}
        response = self.client.post(reverse('rank:login'), data)
        self.assertEqual('1', self.client.session[SESSION_KEY])
        self.assertRedirects(response, reverse('rank:index'))

    def test_redirect(self):
        redirect_url = reverse('rank:index') + '?foo=123&bar=abc'
        login_url = '{}?next={}'.format(reverse('rank:login'), quote(redirect_url))
        response = self.client.get(login_url)
        self.assertContains(response, 'action="{}"'.format(login_url))

        data = {'username': 'alice', 'password': '1234'}
        response = self.client.post(login_url, data)
        self.assertRedirects(response, redirect_url)

    def test_wrong_username_or_password(self):
        data = {'username': 'alice', 'password': '5678'}
        response = self.client.post(reverse('rank:login'), data)
        self.assertTemplateUsed(response, 'rank/login.html')
        self.assertContains(response, '用户名或密码错误')


class RegisterViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def test_get(self):
        response = self.client.get(reverse('rank:register'))
        self.assertTemplateUsed(response, 'rank/register.html')

    def test_invalid_username(self):
        data = {'username': '@#%', 'password': '1234', 'password2': '1234'}
        response = self.client.post(reverse('rank:register'), data)
        self.assertEqual('用户名只能包含字母、数字和下划线', response.context['message'])

    def test_username_already_exists(self):
        data = {'username': 'alice', 'password': '1234', 'password2': '1234'}
        response = self.client.post(reverse('rank:register'), data)
        self.assertEqual('用户名已存在', response.context['message'])

    def test_passwords_not_match(self):
        data = {'username': 'cindy', 'password': '1234', 'password2': '5678'}
        response = self.client.post(reverse('rank:register'), data)
        self.assertEqual('两次密码不一致', response.context['message'])

    def test_ok(self):
        data = {'username': 'bob', 'password': '1234', 'password2': '1234', 'name': '', 'email': ''}
        response = self.client.post(reverse('rank:register'), data)
        self.assertRedirects(response, reverse('rank:login'))
        self.assertTrue(User.objects.filter(username='bob').exists())


class SearchPaperViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def setUp(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})

    @patch('gnnrec.kgrec.recall.recall', return_value=(None, [1, 2]))
    def test_ok(self, recall):
        response = self.client.get(reverse('rank:search-paper'), data={'q': 'xxx'})
        self.assertEqual(200, response.status_code)
        self.assertTemplateUsed(response, 'rank/search_paper.html')
        self.assertQuerysetEqual(response.context['object_list'], ['P1', 'P2'], transform=str)
        recall.assert_called_with(None, 'xxx', settings.PAGE_SIZE)


class PaperDetailViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def setUp(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})

    def test_ok(self):
        response = self.client.get(reverse('rank:paper-detail', args=(1,)))
        self.assertEqual(200, response.status_code)
        self.assertTemplateUsed(response, 'rank/paper_detail.html')
        self.assertContains(response, 'P1')

    def test_not_found(self):
        response = self.client.get(reverse('rank:paper-detail', args=(999,)))
        self.assertEqual(404, response.status_code)


class AuthorDetailViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def setUp(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})

    def test_ok(self):
        response = self.client.get(reverse('rank:author-detail', args=(0,)))
        self.assertEqual(200, response.status_code)
        self.assertTemplateUsed(response, 'rank/author_detail.html')
        self.assertContains(response, 'A0')
        self.assertEqual(3, response.context['n_citation'])
        self.assertQuerysetEqual(response.context['object_list'], ['P0', 'P1'], transform=str)

    def test_not_found(self):
        response = self.client.get(reverse('rank:author-detail', args=(999,)))
        self.assertEqual(404, response.status_code)


class SearchAuthorViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def setUp(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})

    def test_ok(self):
        response = self.client.get(reverse('rank:search-author'), data={'q': 'A0'})
        self.assertEqual(200, response.status_code)
        self.assertTemplateUsed(response, 'rank/search_author.html')
        self.assertQuerysetEqual(response.context['object_list'], ['A0'], transform=str)

    def test_no_result(self):
        response = self.client.get(reverse('rank:search-author'), data={'q': 'xxx'})
        self.assertQuerysetEqual(response.context['object_list'], [], transform=str)
        self.assertContains(response, '未找到学者xxx')


class AuthorRankViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        create_test_data()

    def setUp(self):
        self.client.post(reverse('rank:login'), data={'username': 'alice', 'password': '1234'})

    @patch('gnnrec.kgrec.rank.rank', return_value=(None, [1, 0]))
    def test_ok(self, rank):
        response = self.client.get(reverse('rank:author-rank'), data={'q': 'xxx'})
        self.assertEqual(200, response.status_code)
        self.assertTemplateUsed(response, 'rank/author_rank.html')
        self.assertQuerysetEqual(response.context['object_list'], ['A1', 'A0'], transform=str)
        rank.assert_called_with(None, 'xxx')

    def test_not_login(self):
        self.client.get(reverse('rank:logout'))
        response = self.client.get(reverse('rank:author-rank'), {'q': 'xxx'})
        self.assertRedirects(response, '{}?next={}'.format(
            reverse('rank:login'), quote(reverse('rank:author-rank') + '?q=xxx')
        ))
