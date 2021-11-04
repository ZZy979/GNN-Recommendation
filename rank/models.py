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
    n_citation = models.IntegerField(default=0)

    def __str__(self):
        return self.name


class Paper(models.Model):
    id = models.BigIntegerField(primary_key=True)
    title = models.CharField(max_length=255, db_index=True)
    authors = models.ManyToManyField(Author, related_name='papers', through='Writes')
    venue = models.ForeignKey(Venue, on_delete=models.SET_NULL, null=True)
    year = models.IntegerField()
    abstract = models.CharField(max_length=4095)
    fos = models.ManyToManyField(Field)
    references = models.ManyToManyField('self', related_name='citations', symmetrical=False)
    n_citation = models.IntegerField(default=0)

    def __str__(self):
        return self.title


class Writes(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    order = models.IntegerField(default=1)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['author', 'paper'], name='unique_writes')]
        ordering = ['paper_id', 'order']

    def __str__(self):
        return f'(author_id={self.author_id}, paper_id={self.paper_id}, order={self.order})'
