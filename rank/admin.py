from django.contrib import admin

from .models import Author, Paper, Venue, Institution, Field


class AuthorAdmin(admin.ModelAdmin):
    raw_id_fields = ['institution']


class PaperAdmin(admin.ModelAdmin):
    raw_id_fields = ['authors', 'venue', 'fos', 'references']


admin.site.register(Author, AuthorAdmin)
admin.site.register(Paper, PaperAdmin)
admin.site.register(Venue)
admin.site.register(Institution)
admin.site.register(Field)
