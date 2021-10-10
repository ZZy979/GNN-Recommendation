from django.db import models


class Venue(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, db_index=True)

    def __str__(self):
        return self.name


class Institution(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, db_index=True)

    def __str__(self):
        return self.name


class Field(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Author(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, db_index=True)
    institution = models.ForeignKey(Institution, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.name


class Paper(models.Model):
    id = models.BigIntegerField(primary_key=True)
    title = models.CharField(max_length=255, db_index=True)
    authors = models.ManyToManyField(Author, related_name='papers')
    venue = models.ForeignKey(Venue, on_delete=models.SET_NULL, null=True)
    year = models.IntegerField()
    abstract = models.CharField(max_length=4095)
    fos = models.ManyToManyField(Field)
    references = models.ManyToManyField('self', related_name='citations', symmetrical=False)
    n_citation = models.IntegerField(default=0)

    def __str__(self):
        return self.title
