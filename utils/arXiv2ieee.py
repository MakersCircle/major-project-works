import re


def format_author_list(author_list):
    formated_author_list = []
    for author in author_list:
        parts = author.split()
        name = ''
        for part in parts[:-1]:
            name += f'{part.capitalize()[0]}. '
        name += parts[-1].title()
        formated_author_list.append(name)

    formated_authors = ''
    if len(formated_author_list) == 1:
        formated_authors = formated_author_list[0]
    elif len(formated_author_list) == 2:
        formated_authors = " and ".join(formated_author_list)
    else:
        formated_authors = ", ".join(formated_author_list[:-1]) + ", and " + formated_author_list[-1]

    return formated_authors


def convert_citation(bibtex):
    title_match = re.search(r"title=\{(.*?)\},", bibtex, re.DOTALL)
    author_match = re.search(r"author=\{(.*?)\},", bibtex, re.DOTALL)
    year_match = re.search(r"year=\{(\d{4})\},", bibtex)
    url_match = re.search(r"url=\{(.*?)\},", bibtex)

    if not (title_match and author_match and year_match and url_match):
        raise ValueError("Missing required fields in BibTeX entry.")

    title = title_match.group(1).strip()
    authors = author_match.group(1).strip()
    year = year_match.group(1).strip()
    url = url_match.group(1).strip()

    author_list = authors.split(" and ")

    formated_authors = format_author_list(author_list)\

    ieee_plaintext = f'{formated_authors}, "{title}," {year}, arXiv:{url.split("/")[-1]}'
    ieee_bibitem = f'{formated_authors}, ``{title},\'\' {year}, arXiv:{url.split("/")[-1]}'
    return ieee_plaintext, ieee_bibitem


def main():
    # bibtex = """@misc{wang2023deepaccidentmotionaccidentprediction,
    #       title={DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving},
    #       author={Tianqi Wang and Sukmin Kim and Wenxuan Ji and Enze Xie and Chongjian Ge and Junsong Chen and Zhenguo Li and Ping Luo},
    #       year={2023},
    #       eprint={2304.01168},
    #       archivePrefix={arXiv},
    #       primaryClass={cs.CV},
    #       url={https://arxiv.org/abs/2304.01168},
    # }"""

    bibtex = """@misc{liao2024realtimeaccidentanticipationautonomous,
      title={Real-time Accident Anticipation for Autonomous Driving Through Monocular Depth-Enhanced 3D Modeling}, 
      author={Haicheng Liao and Yongkang Li and Chengyue Wang and Songning Lai and Zhenning Li and Zilin Bian and Jaeyoung Lee and Zhiyong Cui and Guohui Zhang and Chengzhong Xu},
      year={2024},
      eprint={2409.01256},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.01256}, 
}"""
    citation_plaintext, citation_bibitem = convert_citation(bibtex)

    # Generate a unique key for Bibitem (optional: can be based on authors, year, etc.)
    key = ''

    # Print results
    print(f'\nIEEE Plaintext:\n{citation_plaintext}')
    print(f'\nIEEE Bibitem:\n\\bibitem{{{key}}}\n{citation_bibitem}')


if __name__ == '__main__':
    main()