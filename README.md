# Large Language Models for Data Annotation: A Survey

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/nlp24annotation/LLM4Annotation?color=yellow)

- This is a curated list of papers about LLM for Annotation.

- If you want to add new entries, please make PRs with the same format.

- This list serves as a complement to the survey below.

[[Large Language Models for Data Annotation: A Survey]](https://)

<div align=center><img src="https://github.com/nlp24annotation/LLM4Annotation/blob/main/figure/figure.png" width="500" /></div>

If you find this repo helpful, we would appreciate it if you could cite our survey.

## LLM-Based Data Annotation

### Manually Engineered Prompt
- [arXiv 2023] AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators. [[pdf]](https://arxiv.org/pdf/2303.16854.pdf)

- [arXiv 2023] RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. [[pdf]](https://arxiv.org/pdf/2304.06767.pdf)

- [arXiv 2023] Small Models are Valuable Plug-ins for Large Language Models. [[pdf]](https://arxiv.org/pdf/2305.08848.pdf) [[code]](https://github.com/JetRunner/SuperICL)

- [EMNLP 2022] ZeroGen: Efficient Zero-shot Learning via Dataset Generation. [[pdf]](https://arxiv.org/pdf/2202.07922.pdf) [[code]](https://github.com/jiacheng-ye/ZeroGen)

- [NAACL-HLT 2022] Learning To Retrieve Prompts for In-Context Learning. [[pdf]](https://arxiv.org/pdf/2112.08633.pdf) [[code]](https://github.com/OhadRubin/EPR)

- [EMNLP 2021] Constrained Language Models Yield Few-Shot Semantic Parsers. [[pdf]](https://aclanthology.org/2021.emnlp-main.608.pdf) [[code]](https://github.com/microsoft/semantic_parsing_with_constrained_lm)

### Alignment via Pairwise Feedback

- [ACL 2023] Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers. [[pdf]](https://aclanthology.org/2023.findings-acl.247.pdf) [[code]](https://github.com/microsoft/LMOps/tree/main/understand_icl)

- [arXiv 2023] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [[pdf]](https://arxiv.org/pdf/2305.18290.pdf)

- [NeurIPS 2022] Fine-tuning language models to find agreement among humans with diverse preferences. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/f978c8f3b5f399cae464e85f72e28503-Paper-Conference.pdf) 

- [arXiv 2022] Improving alignment of dialogue agents via targeted human judgements. [[pdf]](https://arxiv.org/pdf/2209.14375.pdf) 

- [arXiv 2022] Teaching language models to support answers with verified quotes. [[pdf]](https://arxiv.org/pdf/2203.11147.pdf) [[data]](https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/gophercite-teaching-language-models-to-support-answers-with-verified-quotes/eli5-examples-v2.html)

- [NeurIPS 2020] Learning to summarize with human feedback. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) [[code]](https://github.com/openai/summarize-from-feedback)

- [arXiv 2019] Fine-Tuning Language Models from Human Preferences. [[pdf]](https://arxiv.org/pdf/1909.08593.pdf) [[code]](https://github.com/openai/lm-human-preferences)

<!-- ########################################### -->

## Assessing LLM-Generated Annotations

### Evaluating LLM-Generated Annotations

- [arXiv 2023] AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators. [[pdf]](https://arxiv.org/pdf/2303.16854.pdf)

- [arXiv 2023] Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks. [[pdf]](https://arxiv.org/pdf/2307.02179.pdf)

- [arXiv 2023] Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks. [[pdf]](https://arxiv.org/pdf/2307.02179.pdf)

- [NAACL 2022] LMTurk: Few-Shot Learners as Crowdsourcing Workers in a Language-Model-as-a-Service Framework. [[pdf]](https://aclanthology.org/2022.findings-naacl.51.pdf) [[code]](https://github.com/lmturk)

- [EMNLP 2022] Large Language Models are Few-Shot Clinical Information Extractors. [[pdf]](https://aclanthology.org/2022.emnlp-main.130.pdf) [[data]](https://huggingface.co/datasets/mitclinicalml/clinical-ie)

- [arXiv 2022] Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor. [[pdf]](https://arxiv.org/pdf/2212.09689.pdf) [[code]](https://github.com/orhonovich/unnatural-instructions)

- [arXiv 2020] The Turking Test: Can Language Models Understand Instructions? [[pdf]](https://arxiv.org/pdf/2010.11982.pdf)

### Data Selection via Active Learning

- [EMNLP 2023] FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models [[pdf]](https://aclanthology.org/2023.emnlp-main.896.pdf) [[code]](https://github.com/Justherozen/FreeAL)

- [EMNLP 2023] Active Learning Principles for In-Context Learning with Large Language Models. [[pdf]](https://aclanthology.org/2023.findings-emnlp.334.pdf)

- [IUI 2023] ScatterShot: Interactive In-context Example Curation for Text Transformation. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3581641.3584059) [[code]](https://github.com/tongshuangwu/scattershot)

- [ICML 2023] Prefer to Classify: Improving Text Classifiers via Auxiliary Preference Learning. [[pdf]](https://arxiv.org/pdf/2306.04925) [[code]](https://github.com/minnesotanlp/p2c)

- [arXiv 2023] Large Language Models as Annotators: Enhancing Generalization of NLP Models at Minimal Cost. [[pdf]](https://arxiv.org/pdf/2306.15766.pdf)

- [arXiv 2022] Active learning helps pretrained models learn the intended task. [[pdf]](https://arxiv.org/pdf/2204.08491.pdf) [[code]](https://github.com/alextamkin/active-learning-pretrained-models)

- [EACL 2021] Active Learning for Sequence Tagging with Deep Pre-trained Models and Bayesian Uncertainty Estimates. [[pdf]](https://aclanthology.org/2021.eacl-main.145.pdff)

<!-- ########################################### -->

## Learning with LLM-Generated Annotations

### Target Domain Inference: Direct Utilization of Annotations

- [ECIR 2024] Large Language Models are Zero-Shot Rankers for Recommender Systems. [[pdf]](https://arxiv.org/pdf/2305.08845.pdf) [[code]](https://github.com/RUCAIBox/LLMRank)

- [arXiv 2023] Causal Reasoning and Large Language Models: Opening a New Frontier for Causality. [[pdf]](https://arxiv.org/pdf/2305.00050.pdf) 

- [ACL 2022] An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels. [[pdf]](https://aclanthology.org/2022.acl-long.60.pdf) [[code]](https://github.com/BYU-PCCL/information-theoretic-prompts)

- [TMLR 2022] Emergent Abilities of Large Language Models. [[pdf]](https://arxiv.org/pdf/2206.07682.pdf)

- [NeurIPS 2022] Large Language Models are Zero-Shot Reasoners. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

- [arXiv 2022] Visual Classification via Description from Large Language Models. [[pdf]](https://arxiv.org/pdf/2210.07183.pdf)

- [PMLR 2021] Learning Transferable Visual Models From Natural Language Supervision. [[pdf]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [[code]](https://github.com/OpenAI/CLIP)

- [EMNLP 2019] Language Models as Knowledge Bases? [[pdf]](https://arxiv.org/pdf/1909.01066.pdf) [[code]](https://github.com/facebookresearch/LAMA)

### Knowledge Distillation: Bridging LLM and task-specific models

- [EMNLP 2023] Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. [[pdf]](https://arxiv.org/pdf/2304.09542.pdf) [[code]](https://github.com/sunnweiwei/RankGPT)

- [ACL 2023] Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. [[pdf]](https://arxiv.org/pdf/2305.02301.pdf) [[code]](https://github.com/google-research/distilling-step-by-step)

- [ACL 2023] GPT4All: Training an Assistant-style Chatbot with Large Scale Data Distillation from GPT-3.5-Turbo. [[pdf]](http://static.nomic.ai.s3.amazonaws.com/gpt4all/2023_GPT4All_Technical_Report.pdf) [[code]](https://github.com/nomic-ai/gpt4all)

- [ACL 2023] GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model. [[pdf]](https://aclanthology.org/2023.acl-industry.15.pdf) [[code]](https://github.com/aitsc/GLMKD)

- [EMNLP 2023] Lion: Adversarial Distillation of Proprietary Large Language Models. [[pdf]](https://arxiv.org/pdf/2305.12870.pdf) [[code]](https://github.com/YJiangcm/Lion)

- [arXiv 2023] Specializing Smaller Language Models towards Multi-Step Reasoning. [[pdf]](https://arxiv.org/pdf/2301.12726.pdf)

- [arXiv 2023] Knowledge Distillation of Large Language Models. [[pdf]](https://arxiv.org/pdf/2306.08543.pdf) [[code]](https://github.com/microsoft/LMOps/tree/main/minillm)

- [arXiv 2023] Distilling Large Language Models for Biomedical Knowledge Extraction: A Case Study on Adverse Drug Events. [[pdf]](https://arxiv.org/pdf/2307.06439.pdf)

- [arXiv 2023] Web Content Filtering through knowledge distillation of Large Language Models. [[pdf]](https://arxiv.org/pdf/2305.05027.pdf)

- [ICLR 2022] Knowledge Distillation of Large Language Models. [[pdf]](https://arxiv.org/pdf/2110.08207.pdf) [[code]](https://github.com/bigscience-workshop/t-zero)

- [arXiv 2022] Teaching Small Language Models to Reason. [[pdf]](https://arxiv.org/pdf/2212.08410.pdf)

### Harnessing LLM Annotations for Fine-Tuning and Prompting

#### In-Context Learning (ICL)

- [EMNLP 2023] Active Learning Principles for In-Context Learning with Large Language Models. [[pdf]](https://aclanthology.org/2023.findings-emnlp.334.pdf)

- [ACL 2023] Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models. [[pdf]](https://proceedings.mlr.press/v202/shao23a/shao23a.pdf) 

- [ICLR 2022] Finetuned Language Models Are Zero-Shot Learners. [[pdf]](https://arxiv.org/pdf/2109.01652.pdf) [[code]](https://github.com/google-research/flan)

- [ICLR 2022] Selective Annotation Makes Language Models Better Few-Shot Learners. [[pdf]](https://arxiv.org/pdf/2209.01975.pdf) [[code]](https://github.com/HKUNLP/icl-selective-annotation)

- [NAACL 2022] Improving In-Context Few-Shot Learning via Self-Supervised Training. [[pdf]](https://aclanthology.org/2022.naacl-main.260.pdf) 

- [arXiv 2022] Instruction Induction: From Few Examples to Natural Language Task Descriptions. [[pdf]](https://arxiv.org/pdf/2205.10782.pdf) [[code]](https://github.com/orhonovich/instruction-induction)

- [NeurIPS 2020] Language Models are Few-Shot Learners. [[pdf]](https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

#### Chain-of-Thought Prompting (CoT)

- [ICLR 2023] Automatic chain of thought prompting in large language models. [[pdf]](https://arxiv.org/pdf/2210.03493.pdf) [[code]](https://github.com/amazon-research/auto-cot)

- [ACL 2023] SCOTT: Self-Consistent Chain-of-Thought Distillation. [[pdf]](https://aclanthology.org/2023.acl-long.304v2.pdf)

- [arXiv 2023] Specializing Smaller Language Models towards Multi-Step Reasoning. [[pdf]](https://arxiv.org/pdf/2301.12726.pdf)

- [NeurIPS 2022] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)

- [NeurIPS 2022] Large Language Models are Zero-Shot Reasoners. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

- [arXiv 2022] Rationale-augmented ensembles in language models. [[pdf]](https://arxiv.org/pdf/2207.00747.pdf)

- [ACL 2020] A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers. [[pdf]](https://arxiv.org/pdf/2106.15772.pdf) [[code]](https://github.com/chao-chun/nlu-asdiv-dataset)

- [NAACL 2019] CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. [[pdf]](https://aclanthology.org/N19-1421.pdf) [[code]](https://github.com/jonathanherzig/commonsenseqa)

#### Instruction Tuning (IT)

- [ACL 2023] Crosslingual Generalization through Multitask Finetuning. [[pdf]](https://aclanthology.org/2023.acl-long.891.pdf) [[code]](https://github.com/bigscience-workshop/xmtf)

- [ACL 2023] SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions. [[pdf]](https://aclanthology.org/2023.acl-long.754.pdf) [[code]](https://github.com/yizhongw/self-instruct)

- [ACL 2023] Can Large Language Models Be an Alternative to Human Evaluations? [[pdf]](https://arxiv.org/pdf/2305.01937.pdf)

- [arXiv 2023] LLaMA: Open and Efficient Foundation Language Models. [[pdf]](https://arxiv.org/pdf/2302.13971.pdf)[[code]](https://github.com/facebookresearch/llama)

- [arXiv 2022] Teaching language models to support answers with verified quotes. [[pdf]](https://arxiv.org/pdf/2203.11147.pdf) [[data]](https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/gophercite-teaching-language-models-to-support-answers-with-verified-quotes/eli5-examples-v2.html)

- [arXiv 2022] Scaling instruction-finetuned language models. [[pdf]](https://arxiv.org/pdf/2210.11416.pdf) [[code]](https://huggingface.co/docs/transformers/model_doc/flan-t5)

- [EMNLP 2022] Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. [[pdf]](https://aclanthology.org/2022.emnlp-main.340.pdf) [[code]](https://instructions.apps.allenai.org/)

- [NeurIPS 2020] Language Models are Few-Shot Learners. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

- Stanford alpaca: An instruction-following llama model. [[HTML]](https://crfm.stanford.edu/2023/03/13/alpaca.html) [[code]](https://github.com/tatsu-lab/stanford_alpaca)

#### Alignment Tuning (AT)

- [PMLR 2023] Pretraining Language Models with Human Preferences. [[pdf]](https://proceedings.mlr.press/v202/korbak23a/korbak23a.pdf)[[code]](https://github.com/tomekkorbak/pretraining-with-human-feedback)

- [ICLR 2023] Offline RL for Natural Language Generation with Implicit Language Q Learning. [[pdf]](https://arxiv.org/pdf/2206.11871.pdf) [[code]](https://github.com/Sea-Snell/Implicit-Language-Q-Learning)

- [arXiv 2023] Chain of hindsight aligns language models with feedback. [[pdf]](https://arxiv.org/pdf/2302.02676.pdf)[[code]](https://github.com/lhao499/chain-of-hindsight)

- [arXiv 2023] GPT-4 Technical Report. [[pdf]](https://arxiv.org/pdf/2303.08774.pdf)

- [arXiv 2023] Llama 2: Open Foundation and Fine-Tuned Chat Models. [[pdf]](https://arxiv.org/pdf/2307.09288.pdf) [[code]](https://github.com/facebookresearch/llama)

- [arXiv 2023] RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. [[pdf]](https://arxiv.org/pdf/2309.00267.pdf)

- [NeurIPS 2022] Training language models to follow instructions with human feedback. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)

- [arXiv 2022] Teaching language models to support answers with verified quotes. [[pdf]](https://arxiv.org/pdf/2203.11147.pdf) [[data]](https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/gophercite-teaching-language-models-to-support-answers-with-verified-quotes/eli5-examples-v2.html)


- [arXiv 2019] Fine-Tuning Language Models from Human Preferences. [[pdf]](https://arxiv.org/pdf/1909.08593.pdf)[[code]](https://github.com/openai/lm-human-preferences)

- [arXiv 2019] CTRL: A Conditional Transformer Language Model for Controllable Generation. [[pdf]](https://arxiv.org/pdf/1909.05858.pdf)[[code]](https://github.com/salesforce/ctrl)

- [NeurIPS 2017] Deep Reinforcement Learning from Human Preferences. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)


<!-- ########################################### -->

## Surveys

- [ACM 2023] Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3560815)

- [arXiv 2023] A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models. [[pdf]](https://arxiv.org/pdf/2307.12980.pdf)  [[repo]](https://github.com/JindongGu/Awesome-Prompting-on-Vision-Language-Model/)

- [arXiv 2022] A Survey of Large Language Models. [[pdf]](https://arxiv.org/pdf/2303.18223.pdf) [[repo]](https://github.com/RUCAIBox/LLMSurvey)

- [arXiv 2022] A Survey on In-context Learning. [[pdf]](https://arxiv.org/pdf/2301.00234.pdf)

- [arXiv 2022] A Comprehensive Survey on Instruction Following. [[pdf]](https://arxiv.org/pdf/2303.10475.pdf) [[repo]](https://github.com/RenzeLou/awesome-instruction-learning)


## Toolkits

- LangChain: [[HTML]](https://python.langchain.com/docs/contributing/documentation) [[code]](https://github.com/langchain-ai/langchain)

- Stack AI: [[HTML]](https://www.stack-ai.com/)

- UBIAI: [[HTML]](https://ubiai.tools/)

- Prodigy: [[HTML]](https://prodi.gy/)
