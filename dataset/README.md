# A Dataset of French Trade Directories from the 19th Century for Nested NER task

This dataset is composed of pages and entries extracted from French directories published between 1798 and 1861.

The purpose of this dataset is to evaluate the performance of Nested Named Entity Recognition approaches on 19th century French documents, regarding both clean and noisy texts (due to the OCR step).

![Illustration of the OCR and NER processes](illustrations/overview-intro.svg)

## Source dataset

Our dataset has been build using the following source dataset :

```
N. Abadie, S. Baciocchi, E. Carlinet, J. Chazalon, P. Cristofoli, B. Duménieu and J. Perret, A Dataset of French Trade Directories from the 19th Century (FTD), version 1.0.0, May 2022, online at https://doi.org/10.5281/zenodo.6394464.
```

Details are given [HERE](https://doi.org/10.5281/zenodo.6394464)

## Our experiments // Paper

Details about our experimentson nested NER approaches are given in the follwing paper. [Preprint version can be read on HAL](https://hal.science/hal-03994759v2).

Our code is available on [Git-Hub](https://github.com/soduco/paper-nestedner-icdar23-code).

```
Tual, S., Abadie, N., Chazalon, J., Duménieu, B., & Carlinet, E. (2023). A Benchmark of Nested NER Approaches in Historical Structured Documents. Proceedings of the 17th International Conference on Document Analysis and Recognition, San José, California, USA. 2023. Springer. https://hal.science/hal-03994759v2
```

```bibtex
@inproceedings{nner_benchmark_2023,
	title = {A Benchmark of Nested Named Entity Recognition Approaches in Historical Structured Documents},
    author = {Tual, Solenn and Abadie, Nathalie and Carlinet, Edwin and Chazalon, Joseph and Duménieu, Bertrand},
    booktitle = {Proceedings of the 17th International Conference on Document Analysis and Recognition (ICDAR'23)},
    year = {2023},
    month = aug,
    address = {San José, California, USA},
	  url = {https://hal.science/hal-03994759}
}
```

## Dataset overview

### JSON keys
The following list describes the **keys of the *.JSON*** file which contain the complete materials of our experiments.

- ```id``` : Entry unique ID in a given page
- ```box``` : Bounding box of the entry in the scanned directory page
- ```book``` : Source directory of the entry (*see more information bellow*)
- ```page``` : Page ID in a given directory
- ```valid_box``` : Is the bbox of the entry valid ? (*all bbox are valid here*)
- ```text_ocr_ref``` : OCR extracted and manually corrected text of the entry
- ```nested_ner_xml_ref``` : ```text_ocr_ref``` with nested ner entities
- ```text_ocr_pero``` : OCR extracted text of the entry with PERO-OCR engine (best engine according to *Abadie et al.* experiment)
- ```has_valid_ner_xml_pero``` : Is entities mapping between nested-ner entities annotated by hand on the *ref* text and pero ocr text correct ? (*in our expriments, we only use entries with True value*)
- ```nested_ner_xml_pero``` : Annotated noisy entries produced with PERO OCR
- ```text_ocr_tess``` : OCR extracted text of the entry with Tesseract engine (*not used in our expriments*)
- ```nested_ner_xml_tess``` : Is entities mapping between nested-ner entities annotated by hand on the *ref* text and tesseract text correct ? (*not used in our expriments*)
- ```has_valid_ner_xml_tess``` : Annotated noisy entries produced with Tesseract. (*not used in our expriments*)

### Labels

Our hierachy of entities is a *Part Of* two-levels hierachy. It means that bottom enties are contained in a top level entities.

## Sources documents // Copyright and License
*This section has been written for the [original dataset description](https://zenodo.org/record/6394464).*

The images were extracted from the original source https://gallica.bnf.fr, owned by the *Bibliothèque nationale de France* (French national library).
Original contents from the *Bibliothèque nationale de France* can be reused non-commercially, provided the mention "Source gallica.bnf.fr / Bibliothèque nationale de France" is kept.  
**Researchers do not have to pay any fee for reusing the original contents in research publications or academic works.**  
*Original copyright mentions extracted from https://gallica.bnf.fr/edit/und/conditions-dutilisation-des-contenus-de-gallica on March 29, 2022.*

The original contents were significantly transformed before being included in this dataset.
All derived content is licensed under the permissive *Creative Commons Attribution 4.0 International* license.

Links to original contents

| Directory                     | Page | Link to document                                | Link to page                                               |
| ----------------------------- | ---- | ----------------------------------------------- | ---------------------------------------------------------- |
| Bottin1_1820                  | 107  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624/f108.item   |
| Bottin1_1820                  | 201  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624/f202.item   |
| Bottin1_1820                  | 339  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624/f340.item   |
| Bottin1_1820                  | 589  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624  | https://gallica.bnf.fr/ark:/12148/bpt6k1245624/f590.item   |
| Bottin1_1827                  | 37   | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w/f35.item   |
| Bottin1_1827                  | 117  | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w/f115.item  |
| Bottin1_1827                  | 452  | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w/f450.item  |
| Bottin1_1827                  | 475  | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w | https://gallica.bnf.fr/ark:/12148/bpt6k6292888w/f473.item  |
| Bottin1_1837                  | 80   | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f79.item   |
| Bottin1_1837                  | 114  | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f113.item  |
| Bottin1_1837                  | 203  | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f202.item  |
| Bottin1_1837                  | 345  | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f344.item  |
| Bottin1_1837                  | 663  | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f662.item  |
| Bottin1_1837                  | 667  | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h | https://gallica.bnf.fr/ark:/12148/bpt6k6290660h/f666.item  |
| Bottin3_1854a                 | 72   | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f71.item   |
| Bottin3_1854a                 | 74   | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f73.item   |
| Bottin3_1854a                 | 200  | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f199.item  |
| Bottin3_1854a                 | 238  | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f237.item  |
| Bottin3_1854a                 | 892  | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f891.item  |
| Bottin3_1854a                 | 1049 | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w | https://gallica.bnf.fr/ark:/12148/bpt6k6269398w/f1048.item |
| Cambon_almgene_1841           | 141  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f139.item  |
| Cambon_almgene_1841           | 301  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f299.item  |
| Cambon_almgene_1841           | 330  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f328.item  |
| Cambon_almgene_1841           | 375  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f373.item  |
| Cambon_almgene_1841           | 418  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f416.item  |
| Cambon_almgene_1841           | 487  | https://gallica.bnf.fr/ark:/12148/bpt6k63482147 | https://gallica.bnf.fr/ark:/12148/bpt6k63482147/f485.item  |
| Deflandre_1828                | 278  | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c/f276.item  |
| Deflandre_1828                | 310  | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c/f308.item  |
| Deflandre_1828                | 881  | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c/f879.item  |
| Deflandre_1828                | 937  | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c | https://gallica.bnf.fr/ark:/12148/bpt6k6525634c/f935.item  |
| Deflandre_1829                | 505  | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q/f503.item  |
| Deflandre_1829                | 743  | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q/f741.item  |
| Deflandre_1829                | 949  | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q/f947.item  |
| Deflandre_1829                | 1026 | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q | https://gallica.bnf.fr/ark:/12148/bpt6k6451715q/f1024.item |
| Didot_1841a                   | 162  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f160.item  |
| Didot_1841a                   | 183  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f181.item  |
| Didot_1841a                   | 206  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f204.item  |
| Didot_1841a                   | 316  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f314.item  |
| Didot_1841a                   | 500  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f498.item  |
| Didot_1841a                   | 542  | https://gallica.bnf.fr/ark:/12148/bpt6k62931221 | https://gallica.bnf.fr/ark:/12148/bpt6k62931221/f540.item  |
| Didot_1851a                   | 92   | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f90.item   |
| Didot_1851a                   | 169  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f167.item  |
| Didot_1851a                   | 226  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f224.item  |
| Didot_1851a                   | 415  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f413.item  |
| Didot_1851a                   | 419  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f417.item  |
| Didot_1851a                   | 639  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f637.item  |
| Didot_1851a                   | 698  | https://gallica.bnf.fr/ark:/12148/bpt6k63959929 | https://gallica.bnf.fr/ark:/12148/bpt6k63959929/f696.item  |
| Didot_1854a                   | 83   | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j/f81.item   |
| Didot_1854a                   | 326  | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j/f324.item  |
| Didot_1854a                   | 607  | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j | https://gallica.bnf.fr/ark:/12148/bpt6k6319811j/f605.item  |
| DidotBottin_1860a             | 186  | https://gallica.bnf.fr/ark:/12148/bpt6k63243920 | https://gallica.bnf.fr/ark:/12148/bpt6k63243920/f184.item  |
| DidotBottin_1860a             | 280  | https://gallica.bnf.fr/ark:/12148/bpt6k63243920 | https://gallica.bnf.fr/ark:/12148/bpt6k63243920/f278.item  |
| DidotBottin_1861a             | 238  | https://gallica.bnf.fr/ark:/12148/bpt6k6309075f | https://gallica.bnf.fr/ark:/12148/bpt6k6309075f/f236.item  |
| DidotBottin_1861a             | 424  | https://gallica.bnf.fr/ark:/12148/bpt6k6309075f | https://gallica.bnf.fr/ark:/12148/bpt6k63243920/f422.item  |
| Duverneuil_et_La_Tynna_1801   | 260  | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p/f258.item  |
| Duverneuil_et_La_Tynna_1801   | 371  | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p/f369.item  |
| Duverneuil_et_La_Tynna_1801   | 401  | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p/f399.item  |
| Duverneuil_et_La_Tynna_1801   | 415  | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p/f413.item  |
| Duverneuil_et_La_Tynna_1801   | 454  | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p | https://gallica.bnf.fr/ark:/12148/bpt6k1175057p/f452.item  |
| Duverneuil_et_La_Tynna_1805   | 193  | https://gallica.bnf.fr/ark:/12148/bpt6k62915570 | https://gallica.bnf.fr/ark:/12148/bpt6k62915570/f191.item  |
| Duverneuil_et_La_Tynna_1805   | 250  | https://gallica.bnf.fr/ark:/12148/bpt6k62915570 | https://gallica.bnf.fr/ark:/12148/bpt6k62915570/f248.item  |
| Duverneuil_et_La_Tynna_1805   | 251  | https://gallica.bnf.fr/ark:/12148/bpt6k62915570 | https://gallica.bnf.fr/ark:/12148/bpt6k62915570/f249.item  |
| Duverneuil_et_La_Tynna_1805   | 292  | https://gallica.bnf.fr/ark:/12148/bpt6k62915570 | https://gallica.bnf.fr/ark:/12148/bpt6k62915570/f290.item  |
| Duverneuil_et_La_Tynna_1805   | 305  | https://gallica.bnf.fr/ark:/12148/bpt6k62915570 | https://gallica.bnf.fr/ark:/12148/bpt6k62915570/f303.item  |
| Duverneuil_et_La_Tynna_1806   | 147  | https://gallica.bnf.fr/ark:/12148/bpt6k1245569  | https://gallica.bnf.fr/ark:/12148/bpt6k1245569/f145.item   |
| Duverneuil_et_La_Tynna_1806   | 220  | https://gallica.bnf.fr/ark:/12148/bpt6k1245569  | https://gallica.bnf.fr/ark:/12148/bpt6k1245569/f218.item   |
| Favre_et_Duchesne_1798        | 375  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f373.item  |
| Favre_et_Duchesne_1798        | 429  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f427.item  |
| Favre_et_Duchesne_1798        | 625  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f623.item  |
| Favre_et_Duchesne_1798        | 658  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f656.item  |
| Favre_et_Duchesne_1798        | 700  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f698.item  |
| Favre_et_Duchesne_1798        | 701  | https://gallica.bnf.fr/ark:/12148/bpt6k62929887 | https://gallica.bnf.fr/ark:/12148/bpt6k62929887/f699.item  |
| La_Tynna_1813                 | 163  | https://gallica.bnf.fr/ark:/12148/bpt6k62915718 | https://gallica.bnf.fr/ark:/12148/bpt6k62915718/f164.item  |
| La_Tynna_1813                 | 166  | https://gallica.bnf.fr/ark:/12148/bpt6k62915718 | https://gallica.bnf.fr/ark:/12148/bpt6k62915718/f167.item  |
| La_Tynna_1813                 | 346  | https://gallica.bnf.fr/ark:/12148/bpt6k62915718 | https://gallica.bnf.fr/ark:/12148/bpt6k62915718/f347.item  |
| La_Tynna_1813                 | 377  | https://gallica.bnf.fr/ark:/12148/bpt6k62915718 | https://gallica.bnf.fr/ark:/12148/bpt6k62915718/f378.item  |
| notables_communaux_seine_1801 | 57   | https://gallica.bnf.fr/ark:/12148/bpt6k6417087j | https://gallica.bnf.fr/ark:/12148/bpt6k6417087j/f55.item   |
| notables_communaux_seine_1801 | 144  | https://gallica.bnf.fr/ark:/12148/bpt6k6417087j | https://gallica.bnf.fr/ark:/12148/bpt6k6417087j/f142.item  |

### Versions
- `1.0.0` - 2023-07-20 - Initial release.

