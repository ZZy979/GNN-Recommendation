import os
import tempfile
import zipfile

from gnnrec.kgrec.utils import iter_json


def iter_lines(raw_path, data_type):
    """依次迭代OAG数据集某种类型数据所有txt文件的每一行并将JSON解析为字典

    :param raw_path: str 原始zip文件所在目录
    :param data_type: str 数据类型，author, paper, venue, affiliation之一
    :return: Iterable[dict]
    """
    with tempfile.TemporaryDirectory() as tmp:
        for zip_file in os.listdir(raw_path):
            if zip_file.startswith(f'mag_{data_type}s'):
                with zipfile.ZipFile(os.path.join(raw_path, zip_file)) as z:
                    for txt in z.namelist():
                        print(f'{zip_file}\\{txt}')
                        txt_file = z.extract(txt, tmp)
                        yield from iter_json(txt_file)
                        os.remove(txt_file)
