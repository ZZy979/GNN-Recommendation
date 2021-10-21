from django.apps import AppConfig
from django.conf import settings

from gnnrec.kgrec import recall, rank


class RankConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rank'

    def ready(self):
        if not settings.TESTING:
            from . import views
            views.recall_ctx = recall.get_context(settings.PAPER_EMBEDS_FILE, settings.SCIBERT_MODEL_FILE)
            views.rank_ctx = rank.get_context(views.recall_ctx, settings.AUTHOR_RANK_FILE)
