# Attributions

Most images in this directory are sceptre's own test fixtures. The entries below were
vendored from the `xberg-io/test_documents` corpus (previously consumed as a Git submodule;
see `test_documents/ATTRIBUTIONS.md` and `test_documents/LICENSES.md` upstream) and carry
their own license and attribution obligations, which travel with the files.

The sibling `../ground_truth/` directory holds transcripts of these same images, vendored from the
same upstream corpus. Being derived from the images they describe, they fall under the terms below.

## ndl-lab/pdmocrdataset-part1 (NDL PDM OCR Dataset)

- **Citation:** National Diet Library (Japan), "NDL古典籍OCR / PDM OCR Dataset (tosho_all)", 2022.
- **Source:** https://github.com/ndl-lab/pdmocrdataset-part1 (data: https://lab.ndl.go.jp/dataset/pdm_ocr_dataset/line/tosho_all_linejson.zip)
- **License:** Public Domain Mark 1.0 (source works are copyright-expired; NDL releases openly).
- **Used here:** `ndl_meiji_vertical_01.jpg` .. `ndl_meiji_vertical_05.jpg` — historical Japanese
  printed pages (Meiji-Showa, 1870-1930), predominantly vertical text.

## naver-clova-ix/cord-v2 (CORD)

- **Citation:** Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing",
  NeurIPS 2019 Workshop on Document Intelligence.
- **Source:** https://huggingface.co/datasets/naver-clova-ix/cord-v2
- **License:** CC-BY-4.0
- **Used here:** `cord_receipt_01.jpg` .. `cord_receipt_04.jpg`, from the `test` split.

## TextOCR (Meta)

- **Citation:** Singh et al., "TextOCR: Towards large-scale end-to-end reasoning for
  arbitrary-shaped scene text", CVPR 2021.
- **Source:** https://textvqa.org/textocr/ (images: TextVQA / Open Images;
  https://huggingface.co/datasets/facebook/textvqa)
- **License:** CC-BY-4.0
- **Used here:** `textocr_scene_01.jpg` .. `textocr_scene_03.jpg`; image_ids `e6c1a7b56123bbdb`,
  `76f940b2603a49e7`, `855d76c85603018d`.

## ds4sd/DocLayNet (IBM)

- **Citation:** Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for
  Document-Layout Analysis", KDD 2022.
- **Source:** https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1
- **License:** CDLA-Permissive-1.0
- **Used here:** `doclaynet_page_01.jpg`, `doclaynet_page_02.jpg` — financial-report pages from
  the `test` split (`NASDAQ_ATRI_2003.pdf` p24, `NYSE_MGM_2004.pdf` p49).

## HEIF / AVIF fixtures (libheif-rs)

- **Source:** copied verbatim from the upstream `libheif-rs` repository at
  https://github.com/Cykooz/libheif-rs/tree/master/data.
- **License:** Creative Commons Attribution-ShareAlike 4.0 International
  (https://creativecommons.org/licenses/by-sa/4.0/).
- **Attribution:** Kirill Kuzminykh (Cykooz) and the libheif-rs contributors.
- **Used here:**
  - `alpha.heif` (upstream `alpha.heif`) — HEIF with alpha channel.
  - `test.avif` (upstream `test_nclx.avif`) — AVIF with NCLX color profile.

## Everything else

The remaining images (`balance_sheet_1.png`, `financial_table_1.png`, `invoice_image.png`,
`complex_document*.png`, `ocr_test_*.png`, `layout_parser_ocr.jpg`,
`layout_parser_paper_with_table.jpg`, `chi_sim_image.jpeg`, `jpn_vert.jpeg`, `ocr_image.jpg`,
`test_hello_world.png`, `sample.png`, `sample_text.bmp`, `simple_table.png`,
`Hadley_Crater.jp2`, `english.png`, `french.jpg`, `chinese.jpg`, `japanese.jpg`, `korean.png`,
`cyrillic.png`, `example.png`, `kannada.png`, `telugu.png`) carry no external source recorded in
upstream `test_documents`' own provenance records (`ATTRIBUTIONS.md`, `ground_truth/corpus_manifest.json`,
and git history) and are treated as originally part of the xberg test corpus, same as its
`iwork/` and `hwpx/` fixtures.
