import json
import os
from collections import defaultdict

import scrapy
from itemadapter import ItemAdapter


class ScholarItem(scrapy.Item):
    name = scrapy.Field()
    org = scrapy.Field()
    field = scrapy.Field()
    rank = scrapy.Field()


class AI2000Spider(scrapy.Spider):
    name = 'ai2000'
    allowed_domains = ['aminer.cn']
    custom_settings = {
        'DEFAULT_REQUEST_HEADERS': {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        },
        'DOWNLOAD_DELAY': 20,
        'ITEM_PIPELINES': {'ai2000_crawler.JsonWriterPipeline': 0}
    }

    def __init__(self, save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path

    def start_requests(self):
        return [scrapy.Request(
            'https://apiv2.aminer.cn/magic?a=__mostinfluentialscholars.GetDomainList___',
            callback=self.parse_domain_list, method='POST',
            body='[{"action":"mostinfluentialscholars.GetDomainList","parameters":{"year":2019}}]'
        )]

    def parse_domain_list(self, response):
        domains = json.loads(response.body)['data'][0]['item']
        body_fmt = '[{"action":"ai2000v2.GetDomainTopScholars","parameters":{"year_filter":2020,"domain":"%s","top_n":100,"type":"AI 2000"}}]'
        for domain in domains:
            yield scrapy.Request(
                'https://apiv2.aminer.cn/magic?a=__ai2000v2.GetDomainTopScholars___',
                method='POST', body=body_fmt % domain['id'],
                cb_kwargs={'domain_name': domain['name']}
            )

    def parse(self, response, **kwargs):
        domain_name = kwargs['domain_name']
        scholars = json.loads(response.body)['data'][0]['data']
        for i, scholar in enumerate(scholars[:100]):
            yield ScholarItem(
                name=scholar['person']['name'], org=scholar['org_en'],
                field=domain_name, rank=i
            )


class JsonWriterPipeline:

    def open_spider(self, spider):
        self.scholar_rank = defaultdict(lambda: [None] * 100)
        self.save_path = spider.save_path

    def process_item(self, item, spider):
        scholar = ItemAdapter(item).asdict()
        self.scholar_rank[scholar.pop('field')][scholar.pop('rank')] = scholar
        return item

    def close_spider(self, spider):
        with open(os.path.join(self.save_path), 'w', encoding='utf8') as f:
            json.dump(self.scholar_rank, f, ensure_ascii=False)
