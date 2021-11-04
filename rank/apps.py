from django.apps import AppConfig
from django.conf import settings

from gnnrec.kgrec import recall, rank


class RankConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rank'

    def ready(self):
        if not settings.TESTING:
            from . import views
            views.recall_ctx = recall.get_context()
            views.rank_ctx = rank.get_context(views.recall_ctx)
