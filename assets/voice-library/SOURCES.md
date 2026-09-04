# Voice library sources

本目录的标签只来自素材发布者随数据集提供的元数据；没有通过听感推断种族、性别、口音、年龄或人格。`catalog.json` 是完整来源、许可、文件哈希和改动记录的权威清单。给 Fusion 的目录由后端实时生成，仅保留每条一行的选择必要字段。

- CREMA-D：12 条，来源于 [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)，按仓库提供的说话人 demographic 与文件名情绪标签记录；许可字段按发布仓库列为 [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) / [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)。
- CSTR VCTK 0.92：12 条，来源于 [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443)，说话人性别、年龄、口音和地区来自 corpus metadata，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- Thorsten-Voice：6 条，来源于 [Thorsten-Voice TV-44kHz-Full](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full) 及其发布者样例目录，语言、性别和表演风格采用发布者标签，许可 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)。

超过 5 秒的 8 条素材已裁为 4.8 秒并在末尾做 0.1 秒淡出，没有改变音高或速度；其余条目的具体解码或转存说明逐条记录在 `catalog.json`。运行时会核验目录恰好 30 条、每条实际时长为 2–5 秒、单声道 16-bit PCM WAV，并比对 SHA-256；任一不符即拒绝启动生成链路。
