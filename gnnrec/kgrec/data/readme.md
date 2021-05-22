# Open Academic Graph 2.1数据集
<https://www.aminer.cn/oag-2-1>

使用其中的微软学术(MAG)数据集

## 统计数据
| 类型 | 文件 | 总量 |
| --- | --- | --- |
| author | mag_authors_{0-1}.zip |  |
| paper | mag_papers_{0-16}.zip |  |
| venue | mag_venues.zip | 53422 |
| affiliation | mag_affiliations.zip | 25776 |


## 字段分析：analyze.py
```
数据类型： venue
总量： 53422
最大字段集合： {'JournalId', 'NormalizedName', 'id', 'DisplayName', 'ConferenceId'}
最小字段集合： {'NormalizedName', 'id', 'DisplayName'}
字段出现比例： {'id': 1.0, 'JournalId': 0.9162891692561117, 'DisplayName': 1.0, 'NormalizedName': 1.0, 'ConferenceId': 0.08371083074388828}
示例： {"id": 2898614270, "JournalId": 2898614270, "DisplayName": "Revista de Psiquiatr\u00eda y Salud Mental", "NormalizedName": "revista de psiquiatria y salud mental"}
```

```
数据类型： affiliation
总量： 25776
最大字段集合： {'Latitude', 'Longitude', 'NormalizedName', 'url', 'DisplayName', 'WikiPage', 'id'}
最小字段集合： {'Latitude', 'Longitude', 'NormalizedName', 'DisplayName', 'id'}
字段出现比例： {'id': 1.0, 'DisplayName': 1.0, 'NormalizedName': 1.0, 'WikiPage': 0.9887880198634389, 'Latitude': 1.0, 'Longitude': 1.0, 'url': 0.6649984481688392}
示例： {"id": 3032752892, "DisplayName": "Universidad Internacional de La Rioja", "NormalizedName": "universidad internacional de la rioja", "WikiPage": "https://en.wikipedia.org/wiki/International_University_of_La_Rioja", "Latitude": "42.46270", "Longitude": "2.45500", "url": "https://en.unir.net/"}
```

```
数据类型： author
总量： 243477150
最大字段集合： {'pubs', 'n_pubs', 'normalized_name', 'n_citation', 'id', 'last_known_aff_id', 'name'}
最小字段集合： {'pubs', 'n_pubs', 'normalized_name', 'id', 'name'}
字段出现比例： {'id': 1.0, 'name': 1.0, 'normalized_name': 1.0, 'last_known_aff_id': 0.17816547055853085, 'pubs': 1.0, 'n_pubs': 1.0, 'n_citation': 0.39566894470384595}
示例： {"id": 3040689058, "name": "Jeong Hoe Heo", "normalized_name": "jeong hoe heo", "last_known_aff_id": "59412607", "pubs": [{"i": 2770054759, "r": 10}], "n_pubs": 1, "n_citation": 44}
```

```
数据类型： paper
总量： 240255240
最大字段集合： {'page_end', 'issue', 'url', 'venue', 'references', 'year', 'doc_type', 'publisher', 'volume', 'authors', 'id', 'page_start', 'title', 'n_citation', 'indexed_abstract', 'doi', 'fos'}
最小字段集合： {'volume', 'page_end', 'authors', 'issue', 'id', 'page_start', 'n_citation', 'doc_type', 'publisher'}
字段出现比例： {'id': 1.0, 'title': 0.9999999958377599, 'authors': 1.0, 'venue': 0.5978255167296247, 'year': 0.9999750931550963, 'n_citation': 1.0, 'page_start': 1.0, 'page_end': 1.0, 'doc_type': 1.0, 'publisher': 1.0, 'volume': 1.0, 'issue': 1.0, 'url': 0.9414517743712895, 'doi': 0.37333226530251745, 'indexed_abstract': 0.5832887141192009, 'fos': 0.8758779954185391, 'references': 0.3283648464857624}
示例： {"id": 2507145174, "title": "Structure-Activity Relationships and Kinetic Studies of Peptidic Antagonists of CBX Chromodomains.", "authors": [{"name": "Jacob I. Stuckey", "id": 2277886111, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Catherine Simpson", "id": 2098592917, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Jacqueline L. Norris-Drouin", "id": 2008359561, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Stephanie H. Cholensky", "id": 2280753173, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Junghyun Lee", "id": 2512191121, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Ryan Pasca", "id": 2514369900, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Nancy Cheng", "id": 2500749596, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Bradley M. Dickson", "id": 2144685347, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Kenneth H. Pearce", "id": 2655216619, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Stephen V. Frye", "id": 2155270953, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}, {"name": "Lindsey I. James", "id": 2137194622, "org": "Center for Integrative Chemical Biology and Drug Discovery, Division of Chemical Biology and Medicinal Chemistry, UNC Eshelman School of Pharmacy, University of North Carolina at Chapel Hill , Chapel Hill, North Carolina 27599, United States.\r", "org_id": 114027177}], "venue": {"name": "Journal of Medicinal Chemistry", "id": 162030435}, "year": 2016, "n_citation": 13, "page_start": "8913", "page_end": "8923", "doc_type": "Journal", "publisher": "American Chemical Society", "volume": "59", "issue": "19", "doi": "10.1021/ACS.JMEDCHEM.6B00801", "references": [1976962550, 1982791788, 1988515229, 2000127174, 2002698073, 2025496265, 2032915605, 2050256263, 2059999434, 2076333986, 2077957449, 2082815186, 2105928678, 2116982909, 2120121380, 2146641795, 2149566960, 2156518222, 2160723017, 2170079272, 2207535250, 2270756322, 2326025506, 2327795699, 2332365177, 2346619380, 2466657786], "indexed_abstract": "{\"IndexLength\":108,\"InvertedIndex\":{\"To\":[0],\"better\":[1],\"understand\":[2],\"the\":[3,19,54,70,80,95],\"contribution\":[4],\"of\":[5,21,31,47,56,82,90,98],\"methyl-lysine\":[6],\"(Kme)\":[7],\"binding\":[8,33,96],\"proteins\":[9],\"to\":[10,79],\"various\":[11],\"disease\":[12],\"states,\":[13],\"we\":[14,68],\"recently\":[15],\"developed\":[16],\"and\":[17,36,43,63,73,84],\"reported\":[18],\"discovery\":[20,46],\"1\":[22,48,83],\"(UNC3866),\":[23],\"a\":[24],\"chemical\":[25],\"probe\":[26],\"that\":[27,77],\"targets\":[28],\"two\":[29],\"families\":[30],\"Kme\":[32],\"proteins,\":[34],\"CBX\":[35],\"CDY\":[37],\"chromodomains,\":[38],\"with\":[39,61,101],\"selectivity\":[40],\"for\":[41,87],\"CBX4\":[42],\"-7.\":[44],\"The\":[45],\"was\":[49],\"enabled\":[50],\"in\":[51],\"part\":[52],\"by\":[53,93,105],\"use\":[55],\"molecular\":[57],\"dynamics\":[58],\"simulations\":[59],\"performed\":[60],\"CBX7\":[62,102],\"its\":[64],\"endogenous\":[65],\"substrate.\":[66],\"Herein,\":[67],\"describe\":[69],\"design,\":[71],\"synthesis,\":[72],\"structure\u2013activity\":[74],\"relationship\":[75],\"studies\":[76],\"led\":[78],\"development\":[81],\"provide\":[85],\"support\":[86],\"our\":[88,99],\"model\":[89],\"CBX7\u2013ligand\":[91],\"recognition\":[92],\"examining\":[94],\"kinetics\":[97],\"antagonists\":[100],\"as\":[103],\"determined\":[104],\"surface-plasmon\":[106],\"resonance.\":[107]}}", "fos": [{"name": "chemistry", "w": 0.36301}, {"name": "chemical probe", "w": 0.0}, {"name": "receptor ligand kinetics", "w": 0.46173}, {"name": "dna binding protein", "w": 0.42292}, {"name": "biochemistry", "w": 0.39304}], "url": ["https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.6b00801", "https://www.ncbi.nlm.nih.gov/pubmed/27571219", "http://pubsdc3.acs.org/doi/abs/10.1021/acs.jmedchem.6b00801"]}
```
