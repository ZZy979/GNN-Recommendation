# oag-cs数据集
## 原始数据
[Open Academic Graph 2.1](https://www.aminer.cn/oag-2-1>)

使用其中的微软学术(MAG)数据，总大小169 GB

| 类型 | 文件 | 总量 |
| --- | --- | --- |
| author | mag_authors_{0-1}.zip | 243477150 |
| paper | mag_papers_{0-16}.zip | 240255240 |
| venue | mag_venues.zip | 53422 |
| affiliation | mag_affiliations.zip | 25776 |

## 字段分析
假设原始zip文件所在目录为data/oag/mag/

`python -m gnnrec.kgrec.data.oag.preprocess.analyze {author, paper, vanue, affiliation} data/oag/mag/`

```
数据类型： venue
总量： 53422
最大字段集合： {'JournalId', 'NormalizedName', 'id', 'ConferenceId', 'DisplayName'}
最小字段集合： {'NormalizedName', 'DisplayName', 'id'}
字段出现比例： {'id': 1.0, 'JournalId': 0.9162891692561117, 'DisplayName': 1.0, 'NormalizedName': 1.0, 'ConferenceId': 0.08371083074388828}
示例： {'id': 2898614270, 'JournalId': 2898614270, 'DisplayName': 'Revista de Psiquiatría y Salud Mental', 'NormalizedName': 'revista de psiquiatria y salud mental'}
```

```
数据类型： affiliation
总量： 25776
最大字段集合： {'id', 'NormalizedName', 'url', 'Latitude', 'Longitude', 'WikiPage', 'DisplayName'}
最小字段集合： {'id', 'NormalizedName', 'Latitude', 'Longitude', 'DisplayName'}
字段出现比例： {'id': 1.0, 'DisplayName': 1.0, 'NormalizedName': 1.0, 'WikiPage': 0.9887880198634389, 'Latitude': 1.0, 'Longitude': 1.0, 'url': 0.6649984481688392}
示例： {'id': 3032752892, 'DisplayName': 'Universidad Internacional de La Rioja', 'NormalizedName': 'universidad internacional de la rioja', 'WikiPage': 'https://en.wikipedia.org/wiki/International_University_of_La_Rioja', 'Latitude': '42.46270', 'Longitude': '2.45500', 'url': 'https://en.unir.net/'}
```

```
数据类型： author
总量： 243477150
最大字段集合： {'normalized_name', 'name', 'pubs', 'n_pubs', 'n_citation', 'last_known_aff_id', 'id'}
最小字段集合： {'normalized_name', 'name', 'n_pubs', 'pubs', 'id'}
字段出现比例： {'id': 1.0, 'name': 1.0, 'normalized_name': 1.0, 'last_known_aff_id': 0.17816547055853085, 'pubs': 1.0, 'n_pubs': 1.0, 'n_citation': 0.39566894470384595}
示例： {'id': 3040689058, 'name': 'Jeong Hoe Heo', 'normalized_name': 'jeong hoe heo', 'last_known_aff_id': '59412607', 'pubs': [{'i': 2770054759, 'r': 10}], 'n_pubs': 1, 'n_citation': 44}
```

```
数据类型： paper
总量： 240255240
最大字段集合： {'issue', 'authors', 'page_start', 'publisher', 'doc_type', 'title', 'id', 'doi', 'references', 'volume', 'fos', 'n_citation', 'venue', 'page_end', 'year', 'indexed_abstract', 'url'}
最小字段集合： {'id'}
字段出现比例： {'id': 1.0, 'title': 0.9999999958377599, 'authors': 0.9998381970774082, 'venue': 0.5978255167296247, 'year': 0.9999750931550963, 'page_start': 0.5085962370685443, 'page_end': 0.4468983111460961, 'publisher': 0.5283799512551735, 'issue': 0.41517357124031923, 'url': 0.9414517743712895, 'doi': 0.37333226530251745, 'indexed_abstract': 0.5832887141192009, 'fos': 0.8758779954185391, 'n_citation': 0.3795505812901313, 'doc_type': 0.6272126634990355, 'volume': 0.43235134434528877, 'references': 0.3283648464857624}
示例： {'id': 2507145174, 'title': 'Structure-Activity Relationships and Kinetic Studies of Peptidic Antagonists of CBX Chromodomains.', 'authors': [{'name': 'Jacob I. Stuckey', 'id': 2277886111, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Catherine Simpson', 'id': 2098592917, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Jacqueline L. Norris-Drouin', 'id': 2008359561, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Stephanie H. Cholensky', 'id': 2280753173, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Junghyun Lee', 'id': 2512191121, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Ryan Pasca', 'id': 2514369900, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Nancy Cheng', 'id': 2500749596, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Bradley M. Dickson', 'id': 2144685347, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Kenneth H. Pearce', 'id': 2655216619, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Stephen V. Frye', 'id': 2155270953, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}, {'name': 'Lindsey I. James', 'id': 2137194622, 'org': 'Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r', 'org_id': 114027177}], 'venue': {'name': 'Journal of Medicinal Chemistry', 'id': 162030435}, 'year': 2016, 'n_citation': 13, 'page_start': '8913', 'page_end': '8923', 'doc_type': 'Journal', 'publisher': 'American Chemical Society', 'volume': '59', 'issue': '19', 'doi': '10.1021/ACS.JMEDCHEM.6B00801', 'references': [1976962550, 1982791788, 1988515229, 2000127174, 2002698073, 2025496265, 2032915605, 2050256263, 2059999434, 2076333986, 2077957449, 2082815186, 2105928678, 2116982909, 2120121380, 2146641795, 2149566960, 2156518222, 2160723017, 2170079272, 2207535250, 2270756322, 2326025506, 2327795699, 2332365177, 2346619380, 2466657786], 'indexed_abstract': '{"IndexLength":108,"InvertedIndex":{"To":[0],"better":[1],"understand":[2],"the":[3,19,54,70,80,95],"contribution":[4],"of":[5,21,31,47,56,82,90,98],"methyl-lysine":[6],"(Kme)":[7],"binding":[8,33,96],"proteins":[9],"to":[10,79],"various":[11],"disease":[12],"states,":[13],"we":[14,68],"recently":[15],"developed":[16],"and":[17,36,43,63,73,84],"reported":[18],"discovery":[20,46],"1":[22,48,83],"(UNC3866),":[23],"a":[24],"chemical":[25],"probe":[26],"that":[27,77],"targets":[28],"two":[29],"families":[30],"Kme":[32],"proteins,":[34],"CBX":[35],"CDY":[37],"chromodomains,":[38],"with":[39,61,101],"selectivity":[40],"for":[41,87],"CBX4":[42],"-7.":[44],"The":[45],"was":[49],"enabled":[50],"in":[51],"part":[52],"by":[53,93,105],"use":[55],"molecular":[57],"dynamics":[58],"simulations":[59],"performed":[60],"CBX7":[62,102],"its":[64],"endogenous":[65],"substrate.":[66],"Herein,":[67],"describe":[69],"design,":[71],"synthesis,":[72],"structure–activity":[74],"relationship":[75],"studies":[76],"led":[78],"development":[81],"provide":[85],"support":[86],"our":[88,99],"model":[89],"CBX7–ligand":[91],"recognition":[92],"examining":[94],"kinetics":[97],"antagonists":[100],"as":[103],"determined":[104],"surface-plasmon":[106],"resonance.":[107]}}', 'fos': [{'name': 'chemistry', 'w': 0.36301}, {'name': 'chemical probe', 'w': 0.0}, {'name': 'receptor ligand kinetics', 'w': 0.46173}, {'name': 'dna binding protein', 'w': 0.42292}, {'name': 'biochemistry', 'w': 0.39304}], 'url': ['https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.6b00801', 'https://www.ncbi.nlm.nih.gov/pubmed/27571219', 'http://pubsdc3.acs.org/doi/abs/10.1021/acs.jmedchem.6b00801']}
```

## 第1步：抽取计算机领域的子集
`python -m gnnrec.kgrec.data.oag.preprocess.extract_cs data/oag/mag/ data/oag/cs/`

过滤掉关键字段为空以及标题和摘要过短或过长的论文，
从微软学术抓取了计算机科学下的34个二级领域作为领域字段过滤条件

二级领域列表：[CS_FIELD_L2](oag/config.py)

输出4个文件：

（1）学者：mag_authors.txt

`{"id": aid, "name": "author name", "org": oid}`

（2）论文：mag_papers.txt

```
{
  "id": pid,
  "title": "paper title",
  "authors": [aid],
  "venue": vid,
  "year": publish_year,
  "abstract": "abstract",
  "fos": ["field"],
  "references": [pid]
}
```

（3）期刊：mag_venues.txt

`{"id": vid, "name": "venue name"}`

（4）机构：mag_institutions.txt

`{"id": oid, "name": "org name"}`

## 第2步：预训练论文向量
通过论文二级领域分类任务对预训练的SciBERT模型进行fine-tune，之后将隐藏层输出的128维向量作为paper顶点的输入特征

预训练的SciBERT模型来自Transformers [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased)

1. fine-tune: `python -m gnnrec.kgrec.data.oag.preprocess.fine_tune train data/oag/cs/mag_papers.txt model/scibert.pkl`
2. 推断： `python -m gnnrec.kgrec.data.oag.preprocess.fine_tune infer data/oag/cs/mag_papers.txt model/scibert.pkl data/oag/cs/paper_feat.pkl`

```
Epoch 0 | Train Loss 0.0920 | Train Mirco F1 0.6259 | Val Mirco F1 0.6459
Epoch 1 | Train Loss 0.0660 | Train Mirco F1 0.7247 | Val Mirco F1 0.6558
Epoch 2 | Train Loss 0.0592 | Train Mirco F1 0.7579 | Val Mirco F1 0.6616
Epoch 3 | Train Loss 0.0525 | Train Mirco F1 0.7901 | Val Mirco F1 0.6606
Epoch 4 | Train Loss 0.0468 | Train Mirco F1 0.8170 | Val Mirco F1 0.6585
```

预训练的论文向量保存到paper_feat.pkl文件

## 第3步：构造图数据集
将以上4个txt和1个pkl文件压缩为oag-cs.zip，得到oag-cs数据集的原始数据

将oag-cs.zip文件放到`$DGL_DOWNLOAD_DIR`目录下（环境变量`DGL_DOWNLOAD_DIR`默认为`~/.dgl/`）

```python
from gnnrec.kgrec.data import OAGCSDataset

data = OAGCSDataset()
g = data[0]
```

统计数据见 [OAGCSDataset](oag/cs.py) 的文档字符串

## 下载地址
下载地址：<https://pan.baidu.com/s/1qTth5C_WDxuhJo4yurpITg>，提取码：tz1b

大小：1.38 GB，解压后大小：2.78 GB
