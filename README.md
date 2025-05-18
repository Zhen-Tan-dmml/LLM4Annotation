# Large Language Models for Data Annotation and Synthesis: A Survey

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/Zhen-Tan-dmml/LLM4Annotation?color=yellow)

- This is a curated list of papers about LLM for Data Annotation and Synthesis
    maintained by Dawei Li (daweili5@asu.edu)
- If you want to add new entries, please make PRs with the same format.

- This list serves as a complement to our EMNLP 2024 oral survey:
  [[Large Language Models for Data Annotation and Synthesis: A Survey]](https://arxiv.org/pdf/2402.13446.pdf)

## 🔔 News
- **`2025-4` We update our paper list and include papers for LLM-based data annotation & synthesis in March 2025!**
- **`2025-3` Want to learn more about risks and safety problems of using LLM-based annotation? Check out our new paper list on [AI supervision risk](https://github.com/David-Li0406/AI-Supervision-Risk)!**
- **`2025-3` We update our paper list and include papers for LLM-based data annotation & synthesis in February 2025!**
- **`2025-2` We collect papers and datasets about long-CoT synthesis and distillation, check it!**
- **`2025-2` We update our paper list and include papers for LLM-based data annotation & synthesis in January 2025!**
- **`2025-2` Check our new paper about [preference leakage](https://arxiv.org/abs/2502.01534)!**
- **`2024-12` Check our new paper list and survey on [LLM-as-a-judge](https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge)!**
- **`2024-12` We update our paper list and include papers for LLM-based data annotation & synthesis in December 2024!**

<div align=center><img src="https://github.com/Zhen-Tan-dmml/LLM4Annotation/blob/main/figure/taxonomy.png" width="700" /></div>


<div align=center><img src="https://github.com/Zhen-Tan-dmml/LLM4Annotation/blob/main/figure/framework.png" width="700" /></div>

If you find this repo helpful, we would appreciate it if you could cite our survey.

```
@article{tan2024large,
  title={Large language models for data annotation: A survey},
  author={Tan, Zhen and Li, Dawei and Wang, Song and Beigi, Alimohammad and Jiang, Bohan and Bhattacharjee, Amrita and Karami, Mansooreh and Li, Jundong and Cheng, Lu and Liu, Huan},
  journal={arXiv preprint arXiv:2402.13446},
  year={2024}
}
```

## Long-CoT Synthesis & Distillation

### Papers
- **Self-Training Elicits Concise Reasoning in Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.20122)
- **Rank1: Test-Time Compute for Reranking in Information Retrieval**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.18418)
- **O1 Embedder: Let Retrievers Think Before Action**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07555)
- **MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13383)
- **Small Models Struggle to Learn from Strong Reasoners**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12143)
- **Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.02339)
- **Unveiling the Mechanisms of Explicit CoT Training: How Chain-of-Thought Enhances Reasoning Generalization**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04667)
- **ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06772)
- **s1: Simple test-time scaling**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.19393)
- **Cascaded Self-Evaluation Augmented Training for Efficient Multimodal Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.05662)
- **AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2411.11930)
- **CoT-Valve: Length-Compressible Chain-of-Thought Tuning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.09601)
- **DRT: Deep Reasoning Translation via Long Chain-of-Thought**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.17498)
- **RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems?**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.11284)
- **Demystifying Long Chain-of-Thought Reasoning in LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03373)
- **LIMO: Less is More for Reasoning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03387)
- **ACECODER: Acing Coder RL via Automated Test-Case Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.01718)
- **LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07374)
- **BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03860)
- **Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04404)
- **Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.02508)
- **HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.18925)

### Datasets
- **open-r1/OpenR1-Math-220k**. [[Link]](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- **simplescaling/s1K-1.1**. [[Link]](https://huggingface.co/datasets/simplescaling/s1K-1.1)
- **open-r1/OpenThoughts-114k-math**. [[Link]](https://huggingface.co/datasets/open-r1/OpenThoughts-114k-math)
- **hw-hwei/MedThoughts-8K**. [[Link]](https://huggingface.co/datasets/hw-hwei/MedThoughts-8K)
- **open-thoughts/OpenThoughts-114k**. [[Link]](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- **PrimeIntellect/SYNTHETIC-1-SFT-Data**. [[Link]](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data)
- **FreedomIntelligence/Medical-R1-Distill-Data**. [[Link]](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data)
- **FreedomIntelligence/Medical-R1-Distill-Data-Chinese**. [[Link]](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data-Chinese)
- **Congliu/Chinese-DeepSeek-R1-Distill-data-110k**. [[Link]](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- **NovaSky-AI/Sky-T1_data_17k**. [[Link]](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)
- **bespokelabs/Bespoke-Stratos-17k**. [[Link]](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- **cognitivecomputations/dolphin-r1**. [[Link]](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1)
- **ServiceNow-AI/R1-Distill-SFT**. [[Link]](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
- **Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B**. [[Link]](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)
- **EricLu/SCP-116K**. [[Link]](https://huggingface.co/datasets/EricLu/SCP-116K)
- **PowerInfer/QWQ-LONGCOT-500K**. [[Link]](https://huggingface.co/datasets/PowerInfer/QWQ-LONGCOT-500K)
- **reflex-ai/deepseek-r1-distill-llama-70b-synthetic**. [[Link]](https://huggingface.co/datasets/reflex-ai/deepseek-r1-distill-llama-70b-synthetic)
- **long-cot-dataset collection**. [[Link]](https://huggingface.co/collections/xianbao/long-cot-dataset-676804f4a2d3c09c3e3026d2)

## Updated

### May 2025

- **A Survey on Bridging VLMs and Synthetic Data**. *TechRxiv preprint* (2025) [[Paper]](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174741263.32891073/v1)


### Mar 2025

- **On Limitations of LLM as Annotator for Low Resource Languages**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2411.17637)
- **TeaMs-RL: Teaching LLMs to Generate Better Instruction Datasets via Reinforcement Learning**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2403.08694)
- **Automated Annotation of Evolving Corpora for Augmenting Longitudinal Network Data: A Framework Integrating Large Language Models and Expert Knowledge**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.01672)
- **Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.02718)
- **Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.03261)
- **Memory Is All You Need: Testing How Model Memory Affects LLM Performance in Annotation Tasks**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.04874)
- **HILGEN: Hierarchically-Informed Data Generation for Biomedical NER Using Knowledgebases and Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.04930)
- **AIM-Fair: Advancing Algorithmic Fairness via Selectively Fine-Tuning Biased Models with Contextual Synthetic Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.05665)
- **Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.10662)
- **LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate Speech Moderation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.07237)
- **Word-level Annotation of GDPR Transparency Compliance in Privacy Policies using Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.10727)
- **LEAVS: An LLM-based Labeler for Abdominal CT Supervision**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.13330)
- **Enhancing Arabic Automated Essay Scoring with Synthetic Data and Error Injection**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.17739)
- **Assessing the Reliability and Validity of GPT-4 in Annotating Emotion Appraisal Ratings**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.16883)
- **Protecting Your Video Content: Disrupting Automated Video-based LLM Annotations**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.21824)
- **Data-adaptive Differentially Private Prompt Synthesis for In-Context Learning**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.12085)
- **Tabby: Tabular Data Synthesis with Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.02152)
- **DB-Explore: Automated Database Exploration and Instruction Synthesis for Text-to-SQL**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.04959)
- **Leveraging Large Language Models to Address Data Scarcity in Machine Learning: Applications in Graphene Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.04870)
- **Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.12347)
- **ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.18447)
- **Synthetic Data Generation Using Large Language Models: Advances in Text and Code**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.14023)
- **MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.16212)
- **TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.17195)
- **Unicorn: Text-Only Data Synthesis for Vision Language Model Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.22655)
- **ReSo: A Reward-driven Self-organizing LLM-based Multi-Agent System for Reasoning Tasks**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.02390)
- **From Captions to Rewards (CAREVL): Leveraging Large Language Model Experts for Enhanced Reward Modeling in Large Vision-Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.06260)
- **Cockatiel: Ensembling Synthetic and Human Preferenced Training for Detailed Video Caption**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.09279)
- **MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.16096)
- **From Annotation to Adaptation: Metrics, Synthetic Data, and Aspect Extraction for Aspect-Based Sentiment Analysis with Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.20715)
- **Quality-Driven Curation of Remote Sensing Vision-Language Data via Learned Scoring Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.00743)
- **KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.02951)
- **Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.07826)
- **MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.09499)
- **DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.12797)
- **Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.08741)
- **CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.01836)
- **Targeted Distillation for Sentiment Analysis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.03225)
- **Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.16385)
- **Towards Understanding Distilled Reasoning Models: A Representational Approach**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.03730)
- **1.4 Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.19633)
- **Scaling Laws of Synthetic Data for Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.19551)
- **ELTEX: A Framework for Domain-Driven Synthetic Data Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.15055)
- **Leveraging large language models to examine the interaction between investor sentiment and stock performance**. *ArXiv preprint* (2025) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0952197625006025)
- **Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.17928)
- **TREESYNTH: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.17195)
- **Machine Learning Approaches in Software Vulnerability Detection: A Systematic Review and Analysis of Contemporary Methods**. *ArXiv preprint* (2025) [[Paper]](https://www.researchsquare.com/article/rs-5975490/latest)
- **Empowering Time Series Analysis with Synthetic Data: A Survey and Outlook in the Era of Foundation Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.11411)
- **Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.11096)
- **Generative AI in Transportation Planning: A Survey**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.07158)
- **Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.06778)
- **A large language model-enabled machining process knowledge graph construction method for intelligent process planning**. *ArXiv preprint* (2025) [[Paper]](https://www.sciencedirect.com/science/article/pii/S1474034625001375)
- **Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.05587)
- **PromptPex: Automatic Test Generation for Language Model Prompts**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2503.05070)


### Feb 2025

- **SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12025)
- **Stronger Models are NOT Stronger Teachers for Instruction Tuning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2411.07133)
- **Preference Leakage: A Contamination Problem in LLM-as-a-judge**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.01534)
- **AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.01977)
- **Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.01991)
- **Can Large Language Models Capture Video Game Engagement?**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04379)
- **BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03860)
- **Great Models Think Alike and this Undermines AI Oversight**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04313)
- **Self-seeding and Multi-intent Self-instructing LLMs for Generating Intent-aware Information-Seeking dialogs**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2402.11633)
- **SynthDetoxM: Modern LLMs are Few-Shot Parallel Detoxification Data Annotators**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06394)
- **VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06737)
- **Perceived Confidence Scoring for Data Annotation with Zero-Shot LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07186)
- **NARCE: A Mamba-Based Neural Algorithmic Reasoner Framework for Online Complex Event Detection**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07250)
- **Semantic to Structure: Learning Structural Representations for Infringement Detection**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07323)
- **Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.05400)
- **PropaInsight: Toward Deeper Understanding of Propaganda in Terms of Techniques, Appeals, and Intent**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2409.18997)
- **Large Language Models and Synthetic Data for Monitoring Dataset Mentions in Research Papers**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.10263)
- **Batch-Adaptive Annotations for Causal Inference with Complex-Embedded Outcomes**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.10605)
- **PlagBench: Exploring the Duality of Large Language Models in Plagiarism Generation and Detection**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2406.16288)
- **Grammatical Error Correction for Low-Resource Languages: The Case of Zarma**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.15539)
- **Code Simulation as a Proxy for High-order Tasks in Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03568)
- **Scaling Autonomous Agents via Automatic Reward Modeling And Planning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12130)
- **Efficient Machine Translation Corpus Generation: Integrating Human-in-the-Loop Post-Editing with Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12755)
- **Synthetic Data Generation for Culturally Nuanced Commonsense Reasoning in Low-Resource Languages**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12932)
- **Do we still need Human Annotators? Prompting Large Language Models for Aspect Sentiment Quad Prediction**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13044)
- **Theorem Prover as a Judge for Synthetic Data Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13137)
- **Template-Based Visual Program Distillation**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.08564)
- **SPPD: Self-training with Process Preference Learning Using Dynamic Value Margin**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13516)
- **Don't Stop the Multi-Party! On Generating Synthetic Multi-Party Conversations with Constraints**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13592)
- **How to Get Your LLM to Generate Challenging Problems for Evaluation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14678)
- **Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14860)
- **Data-Constrained Synthesis of Training Data for De-Identification**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14677)
- **Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.15592)
- **Instruction-Tuning LLMs for Event Extraction with Annotation Guidelines**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.16377)
- **Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14948)
- **On Synthetic Data Strategies for Domain-Specific Generative Retrieval**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.17957)
- **Are Chatbots Reliable Text Annotators? Sometimes**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2311.05769)
- **Accelerating Unbiased LLM Evaluation via Synthetic Feedback**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.10563)
- **MathClean: A Benchmark for Synthetic Mathematical Data Cleaning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19058)
- **Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19545)
- **Old Experience Helps: Leveraging Survey Methodology to Improve AI Text Annotation Reliability in Social Sciences**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19679)
- **Few-Shot Multilingual Open-Domain QA from 5 Examples**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19722)
- **CoddLLM: Empowering Large Language Models for Data Analytics**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.00329)
- **Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.00955)
- **Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.15606)
- **Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03843)
- **AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2408.00764)
- **CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03997)
- **ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.05567)
- **TF-DCon: Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2310.09874)
- **Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07045)
- **Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07942)
- **O1 Embedder: Let Retrievers Think Before Action**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.07555)
- **COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2408.05006)
- **Beyond Sample-Level Feedback: Using Reference-Level Feedback to Guide Data Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04511)
- **Few-shot LLM Synthetic Data with Distribution Matching**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.08661)
- **Improve LLM-as-a-Judge Ability as a General Ability**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.11689)
- **You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13001)
- **MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13383)
- **Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14523)
- **Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.16457)
- **WildLong: Synthesizing Realistic Long-Context Instruction Data at Scale**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.16684)
- **AutoLogi: Automated Generation of Logic Puzzles for Evaluating Reasoning Abilities of Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.16906)
- **Mutual Reinforcement of LLM Dialogue Synthesis and Summarization Capabilities for Few-Shot Dialogue Summarization**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.17328)
- **Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic: An Approach Based on Error Tagging Model and Synthetic Data Generating Model**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.05312)
- **Simulation as Reality? The Effectiveness of LLM-Generated Data in Open-ended Question Assessment**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06371)
- **Automatic Annotation Augmentation Boosts Translation between Molecules and Natural Language**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06634)
- **Rationalization Models for Text-to-SQL**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06759)
- **Organize the Web: Constructing Domains Enhances Pre-Training Data Curation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.10341)
- **Improving Scientific Document Retrieval with Concept Coverage-based Query Set Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.11181)
- **Stackelberg Game Preference Optimization for Data-Efficient Alignment of Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.18099)
- **mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.08468)
- **SynthVLM: High-Efficiency and High-Quality Synthetic Data for Vision Language Models**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2407.20756)
- **VividMed: Vision Language Model with Versatile Visual Grounding for Medicine**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.12694)
- **OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.11102)
- **RewardDS: Privacy-Preserving Fine-Tuning for Large Language Models via Reward Driven Data Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.18517)
- **RankFlow: A Multi-Role Collaborative Reranking Workflow Utilizing Large Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.00709)
- **Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.02988)
- **Knowledge Distillation from Large Language Models for Household Energy Modeling**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03034)
- **KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.05223)
- **DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Disease**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04394)
- **The Best Instruction-Tuning Data are Those That Fit**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04194)
- **Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2411.13045)
- **Who Taught You That? Tracing Teachers in Model Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.06659)
- **Demystifying Domain-adaptive Post-training for Financial LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.04961)
- **RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.09183)
- **Syntriever: How to Train Your Retriever with Synthetic Data from LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.03824)
- **Efficient Multitask Learning in Small Language Models Through Upside-Down Reinforcement Learning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.09854)
- **Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.11191)
- **ShieldLearner: A New Paradigm for Jailbreak Attack Defense in LLMs**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13162)
- **Quantification of Large Language Model Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.12619)
- **Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.11598)
- **Self-Enhanced Reasoning Training: Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12744)
- **Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14905)
- **Lean-ing on Quality: How High-Quality Data Beats Diverse Multilingual Data in AutoFormalization**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.15795)
- **Proxona: Supporting Creators' Sensemaking and Ideation with LLM-Powered Audience Personas**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2408.10937)
- **The Self-Improvement Paradox: Can Language Models Bootstrap Reasoning Capabilities without External Scaffolding?**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13441)
- **ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.13458)
- **Learning from Committee: Reasoning Distillation from a Mixture of Teachers with Peer-Review**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.03663)
- **Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.14272)
- **Enhancing Domain-Specific Retrieval-Augmented Generation: Synthetic Data Generation and Evaluation using Reasoning Models**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.15854)
- **PPC-GPT: Federated Task-Specific Compression of Large Language Models via Pruning and Chain-of-Thought Distillation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.15857)
- **Learning with Less: Knowledge Distillation from Large Language Models via Unlabeled Data**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2411.08028)
- **Small Models Struggle to Learn from Strong Reasoners**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.12143)
- **HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.01524)
- **Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.18001)
- **Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.02674)
- **SuperCorrect: Advancing Small LLM Reasoning with Thought Template Distillation and Self-Correction**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.09008)
- **Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones?**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19557)
- **Teaching Dense Retrieval Models to Specialize with Listwise Distillation and LLM Data Augmentation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.19712)
- **TinyThinker: Distilling Reasoning through Coarse-to-Fine Knowledge Internalization with Self-Reflection**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.08024)
- **Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.02339)
- **Escaping Collapse: The Strength of Weak Data for Large Language Model Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.08924)
- **Measuring Diversity in Synthetic Datasets**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.08512)
- **ARISE: Iterative Rule Induction and Synthetic Data Generation for Text Classification**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.05923)
- **Understanding and Mitigating the Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.04419)
- **Synthetic Artifact Auditing: Tracing LLM-Generated Synthetic Data Usage in Downstream Applications**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.00808)


### Jan 2025

- **Distilling Desired Comments for Enhanced Code Review with Large Language Models**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.20340)
- **Evaluating Large Language Models Against Human Annotators in Latent Content Analysis: Sentiment, Political Leaning, Emotional Intensity, and Sarcasm**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.02532)
- **Biomedical Relation Extraction via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.05155)
- **Enabling Scalable Oversight via Self-Evolving Critic**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.05727)
- **Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.09004)
- **Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.10893)
- **Leveraging Graph Structures and Large Language Models for End-to-End Synthetic Task-Oriented Dialogues**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.11977)
- **The Promises and Pitfalls of LLM Annotations in Dataset Labeling: a Case Study on Media Bias Detection**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2411.11081)
- **Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.15108)
- **SyntheT2C: Generating Synthetic Data for Fine-Tuning Large Language Models on the Text2Cypher Task**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2406.10710)
- **PISCO: Pretty Simple Compression for Retrieval-Augmented Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.16075)
- **FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.17144)
- **RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.18056)
- **Leveraging Sparsity for Sample-Efficient Preference Learning: A Theoretical Perspective**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.18282)
- **ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2405.17743)
- **TabuLa: Harnessing Language Models for Tabular Data Synthesis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2310.12746)
- **CDS: Data Synthesis Method Guided by Cognitive Diagnosis Theory**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.07674)
- **Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.11759)
- **Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.14002)
- **URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.04686)
- **Scaling Large Vision-Language Models for Enhanced Multimodal Comprehension In Biomedical Image Analysis**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.15370)
- **OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.15427)
- **KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2403.03101)
- **SampleLLM: Optimizing Tabular Data Synthesis in Recommendations**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.16125)
- **Augmenting Human-Annotated Training Data with Large Language Model Generation and Distillation in Open-Response Assessment**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.09126)
- **Descriptive Caption Enhancement with Visual Specialists for Multimodal Perception**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2412.14233)
- **Embedding-Driven Diversity Sampling to Improve Few-Shot Synthetic Data Generation**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.11199)
- **Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2410.18558)
- **EPIC: Effective Prompting for Imbalanced-Class Data Synthesis in Tabular Data Classification via Large Language Models**. *ArXiv preprint* (2024) [[Paper]](https://arxiv.org/abs/2404.12404)
- **Contrastive Private Data Synthesis via Weighted Multi-PLM Fusion**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2502.00245)
- **Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.12789)
- **Large Language Models for Synthetic Dataset Generation: A Case Study on Ethereum Smart Contract DoS Vulnerabilities**. *ArXiv preprint* (2025) [[Paper]](https://www.researchgate.net/profile/Giuseppe-Destefanis-2/publication/388273070_Large_Language_Models_for_Synthetic_Dataset_Generation_A_Case_Study_on_Ethereum_Smart_Contract_DoS_Vulnerabilities/links/67910e39ec3ae3435a7590b8/Large-Language-Models-for-Synthetic-Dataset-Generation-A-Case-Study-on-Ethereum-Smart-Contract-DoS-Vulnerabilities.pdf)
- **On evaluation protocols for data augmentation in a limited data scenario**. *ArXiv preprint* (2025) [[Paper]](https://aclanthology.org/2025.coling-main.231/)
- **WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training**. *ArXiv preprint* (2025) [[Paper]](https://arxiv.org/abs/2501.18511)
- **Synthetic Data Generation Using Large Language Models for Financial Question Answering**. *ArXiv preprint* (2025) [[Paper]](https://aclanthology.org/2025.finnlp-1.7/)
- **SKIntern: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models**. *ArXiv preprint* (2025) [[Paper]](https://aclanthology.org/2025.coling-main.215/)


### Dec 2024

- **The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators** Tzu-Heng Huang, Catherine Cao, Vaishnavi Bhargava, Frederic Sala. *Advances in Neural Information Processing Systems (2024, Spotlight)* [[link]](https://arxiv.org/abs/2407.11004) 

- **A text-to-tabular approach to generate synthetic patient data using LLMs** Margaux Tornqvist, Jean-Daniel Zucker, Tristan Fauvel, Nicolas Lambert, Mathilde Berthelot, Antoine Movschin. *arXiv preprint arXiv:2412.05153* (2024) [[link]](https://arxiv.org/pdf/2412.05153)

- **Give me Some Hard Questions: Synthetic Data Generation for Clinical QA** Fan Bai, Keith Harrigian, Joel Stremmel, Hamid Hassanzadeh, Ardavan Saeedi, Mark Dredze. *arXiv preprint arXiv:2412.04573* (2024) [[link]](https://arxiv.org/pdf/2412.04573)

- **Building a Family of Data Augmentation Models for Low-cost LLM Fine-tuning on the Cloud** Yuanhao Yue, Chengyu Wang, Jun Huang, Peng Wang. *arXiv preprint arXiv:2412.04871* (2024) [[link]](https://arxiv.org/pdf/2412.04871)

- **Can Open-source LLMs Enhance Data Synthesis for Toxic Detection?: An Experimental Study** Zheng Hui, Zhaoxiao Guo, Hang Zhao, Juanyong Duan, Lin Ai, Yinheng Li, Julia Hirschberg, Congrui Huang. *arXiv preprint arXiv:2411.15175* (2024) [[link]](https://arxiv.org/pdf/2411.15175)

- **Evaluating Large Language Model Capability in Vietnamese Fact-Checking Data Generation** Long Truong To, Hung Tuan Le, Dat Van-Thanh Nguyen, Manh Trong Nguyen, Tri Thien Nguyen, Tin Van Huynh, Kiet Van Nguyen. *arXiv preprint arXiv:2411.05641* (2024) [[link]](https://arxiv.org/pdf/2411.05641)

- **CorrSynth -- A Correlated Sampling Method for Diverse Dataset Generation from LLMs** Suhas S Kowshik, Abhishek Divekar, Vijit Malik. *arXiv preprint arXiv:2411.08553* (2024) [[link]](https://arxiv.org/pdf/2411.08553)

- **AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials** Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, Tao Yu. *arXiv preprint arXiv:2412.09605* (2024) [[link]](https://arxiv.org/pdf/2412.09605)

- **Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel** Zun Wang, Jialu Li, Yicong Hong, Songze Li, Kunchang Li, Shoubin Yu, Yi Wang, Yu Qiao, Yali Wang, Mohit Bansal, Limin Wang. *arXiv preprint arXiv:2412.08467* (2024) [[link]](https://arxiv.org/pdf/2412.08467)

- **Filling Memory Gaps: Enhancing Continual Semantic Parsing via SQL Syntax Variance-Guided LLMs without Real Data Replay** Ruiheng Liu, Jinyu Zhang, Yanqi Song, Yu Zhang, Bailong Yang. *arXiv preprint arXiv:2412.07246* (2024) [[link]](https://arxiv.org/pdf/2412.07246)

- **Language Models as Continuous Self-Evolving Data Engineers** Peidong Wang, Ming Wang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang. *arXiv preprint arXiv:2412.15151* (2024) [[link]](https://arxiv.org/pdf/2412.15151)

- **From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research** Xiang Cheng, Raveesh Mayya, João Sedoc. *arXiv preprint arXiv:2412.14461* (2024) [[link]](https://arxiv.org/pdf/2412.14461)

- **Cognition Chain for Explainable Psychological Stress Detection on Social Media** Xin Wang, Boyan Gao, Yi Dai, Lei Cao, Liang Zhao, Yibo Yang, David Clifton. *arXiv preprint arXiv:2412.14009* (2024) [[link]](https://arxiv.org/pdf/2412.14009)

- **OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain** Shuting Wang, Jiejun Tan, Zhicheng Dou, Ji-Rong Wen. *arXiv preprint arXiv:2412.13018* (2024) [[link]](https://arxiv.org/pdf/2412.13018)

- **DS2-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot Aspect-Based Sentiment Analysis** Hongling Xu, Yice Zhang, Qianlong Wang, Ruifeng Xu. *arXiv preprint arXiv:2412.14849* (2024) [[link]](https://arxiv.org/pdf/2412.14849)

- **MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval** Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, Yongping Xiong. *arXiv preprint arXiv:2412.14475* (2024) [[link]](https://arxiv.org/pdf/2412.14475)

- **Can LLMs Convert Graphs to Text-Attributed Graphs?** Zehong Wang, Sidney Liu, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye. *arXiv preprint arXiv:2412.10136* (2024) [[link]](https://arxiv.org/pdf/2412.10136)

- **A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Reasoning Instructions** Jiankang Wang, Jianjun Xu, Xiaorui Wang, Yuxin Wang, Mengting Xing, Shancheng Fang, Zhineng Chen, Hongtao Xie, Yongdong Zhang. *arXiv preprint arXiv:2412.08864* (2024) [[link]](https://arxiv.org/pdf/2412.08864)

- **CoPrUS: Consistency Preserving Utterance Synthesis towards more realistic benchmark dialogues** Sebastian Steindl, Ulrich Schäfer, Bernd Ludwig. *arXiv preprint arXiv:2412.07515* (2024) [[link]](https://arxiv.org/pdf/2412.07515)

- **FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering** Amirhossein Abaskohi, Spandana Gella, Giuseppe Carenini, Issam H. Laradji. *arXiv preprint arXiv:2412.07030* (2024) [[link]](https://arxiv.org/pdf/2412.07030)

- **AIDE: Task-Specific Fine Tuning with Attribute Guided Multi-Hop Data Expansion** Jiayu Li, Xuan Zhu, Fang Liu, Yanjun Qi. *arXiv preprint arXiv:2412.06136* (2024) [[link]](https://arxiv.org/pdf/2412.06136)

- **Seed-CTS: Unleashing the Power of Tree Search for Superior Performance in Competitive Coding Tasks** Hao Wang, Boyi Liu, Yufeng Zhang, Jie Chen. *arXiv preprint arXiv:2412.12544* (2024) [[link]](https://arxiv.org/pdf/2412.12544)

- **Text2Relight: Creative Portrait Relighting with Text Guidance** Junuk Cha, Mengwei Ren, Krishna Kumar Singh, He Zhang, Yannick Hold-Geoffroy, Seunghyun Yoon, HyunJoon Jung, Jae Shin Yoon, Seungryul Baek. *arXiv preprint arXiv:2412.13734* (2024) [[link]](https://arxiv.org/pdf/2412.13734)

- **ResoFilter: Fine-grained Synthetic Data Filtering for Large Language Models through Data-Parameter Resonance Analysis** Zeao Tu, Xiangdi Meng, Yu He, Zihan Yao, Tianyu Qi, Jun Liu, Ming Li. *arXiv preprint arXiv:2412.14809* (2024) [[link]](https://arxiv.org/pdf/2412.14809)

- **How to Synthesize Text Data without Model Collapse?** Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, Bowen Zhou. *arXiv preprint arXiv:2412.14689* (2024) [[link]](https://arxiv.org/pdf/2412.14689)

- **Libri2Vox Dataset: Target Speaker Extraction with Diverse Speaker Conditions and Synthetic Data** Yun Liu, Xuechen Liu, Xiaoxiao Miao, Junichi Yamagishi. *arXiv preprint arXiv:2412.12512* (2024) [[link]](https://arxiv.org/pdf/2412.12512)

- **ALMA: Alignment with Minimal Annotation** Michihiro Yasunaga, Leonid Shamis, Chunting Zhou, Andrew Cohen, Jason Weston, Luke Zettlemoyer, Marjan Ghazvininejad. *arXiv preprint arXiv:2412.04305* (2024) [[link]](https://arxiv.org/pdf/2412.04305)

- **MAG-V: A Multi-Agent Framework for Synthetic Data Generation and Verification** Saptarshi Sengupta, Kristal Curtis, Akshay Mallipeddi, Abhinav Mathur, Joseph Ross, Liang Gou. *arXiv preprint arXiv:2412.04494* (2024) [[link]](https://arxiv.org/pdf/2412.04494)

- **Piecing It All Together: Verifying Multi-Hop Multimodal Claims** Haoran Wang, Aman Rangapur, Xiongxiao Xu, Yueqing Liang, Haroon Gharwi, Carl Yang, Kai Shu. *arXiv preprint arXiv:2411.09547* (2024) [[link]](https://arxiv.org/pdf/2411.09547)

- **Automated Collection of Evaluation Dataset for Semantic Search in Low-Resource Domain Language** Anastasia Zhukova, Christian E. Matt, Bela Gipp. *arXiv preprint arXiv:2412.10008* (2024) [[link]](https://arxiv.org/pdf/2412.10008)

- **Argumentative Experience: Reducing Confirmation Bias on Controversial Issues through LLM-Generated Multi-Persona Debates** Li Shi, Houjiang Liu, Yian Wong, Utkarsh Mujumdar, Dan Zhang, Jacek Gwizdka, Matthew Lease. *arXiv preprint arXiv:2412.04629* (2024) [[link]](https://arxiv.org/pdf/2412.04629)

- **A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI** Beiduo Chen, Siyao Peng, Anna Korhonen, Barbara Plank. *arXiv preprint arXiv:2412.13942* (2024) [[link]](https://arxiv.org/pdf/2412.13942)

- **On Limitations of LLM as Annotator for Low Resource Languages** Suramya Jadhav, Abhay Shanbhag, Amogh Thakurdesai, Ridhima Sinare, Raviraj Joshi. *arXiv preprint arXiv:2411.17637* (2024) [[link]](https://arxiv.org/pdf/2411.17637)

- **LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification** Taja Kuzman, Nikola Ljubešić. *arXiv preprint arXiv:2411.19638* (2024) [[link]](https://arxiv.org/pdf/2411.19638)

- **DSAI: Unbiased and Interpretable Latent Feature Extraction for Data-Centric AI** Hyowon Cho, Soonwon Ka, Daechul Park, Jaewook Kang, Minjoon Seo, Bokyung Son. *arXiv preprint arXiv:2412.06303* (2024) [[link]](https://arxiv.org/pdf/2412.06303)

- **Rethinking Emotion Annotations in the Era of Large Language Models** Minxue Niu, Yara El-Tawil, Amrit Romana, Emily Mower Provost. *arXiv preprint arXiv:2412.07906* (2024) [[link]](https://arxiv.org/pdf/2412.07906)

- **DialogAgent: An Auto-engagement Agent for Code Question Answering Data Production** Xiaoyun Liang, Jingyi Ren, Jiayi Qi, Chao Peng, Bo Jiang. *arXiv preprint arXiv:2412.08069* (2024) [[link]](https://arxiv.org/pdf/2412.08069)

- **Automated Collection of Evaluation Dataset for Semantic Search in Low-Resource Domain Language** Anastasia Zhukova, Christian E. Matt, Bela Gipp. *arXiv preprint arXiv:2412.10008* (2024) [[link]](https://arxiv.org/pdf/2412.10008)

- **Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments** Tuka Alhanai, Adam Kasumovic, Mohammad Ghassemi, Aven Zitzelberger, Jessica Lundin, Guillaume Chabot-Couture. *arXiv preprint arXiv:2412.12417* (2024) [[link]](https://arxiv.org/pdf/2412.12417)

- **Enhancing Persona Classification in Dialogue Systems: A Graph Neural Network Approach** Konstantin Zaitsev. *arXiv preprint arXiv:2412.13283* (2024) [[link]](https://arxiv.org/pdf/2412.13283)

- **Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data** Shaina Raza, Drai Paulen-Patterson, Chen Ding. *arXiv preprint arXiv:2412.14276* (2024) [[link]](https://arxiv.org/pdf/2412.14276)

- **VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval** Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, Yongping Xiong. *arXiv preprint arXiv:2406.04292* (2024) [[link]](https://arxiv.org/pdf/2406.04292)

## LLM-Based Data Annotation

### Instruction & Response

- **Generating training data with language models: Towards zero-shot language understanding.** Meng, Yu, Huang, Jiaxin, Zhang, Yu, and Han, Jiawei. *Advances in Neural Information Processing Systems* (2022) [[link]](https://proceedings.neurips.cc/paper_files/paper/2022/file/0346c148ba1c21c6b4780a961ea141dc-Paper-Conference.pdf)

- **ZeroGen: Efficient Zero-shot Learning via Dataset Generation.** Ye, Jiacheng, Gao, Jiahui, Li, Qintong, Xu, Hang, Feng, Jiangtao, Wu, Zhiyong, Yu, Tao, and Kong, Lingpeng. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (2022) [[link]](https://arxiv.org/pdf/2202.07922)

- **CodecLM: Aligning Language Models with Tailored Synthetic Data.** Wang, Zifeng, Li, Chun-Liang, Perot, Vincent, Le, Long T, Miao, Jin, Zhang, Zizhao, Lee, Chen-Yu, and Pfister, Tomas. *arXiv preprint arXiv:2404.05875* (2024) [[link]](https://arxiv.org/pdf/2404.05875)

- **UniGen: A Unified Framework for Textual Dataset Generation Using Large Language Models.** Wu, Siyuan, Huang, Yue, Gao, Chujie, Chen, Dongping, Zhang, Qihui, Wan, Yao, Zhou, Tianyi, Zhang, Xiangliang, Gao, Jianfeng, Xiao, Chaowei, and others. *arXiv preprint arXiv:2406.18966* (2024) [[link]](https://arxiv.org/pdf/2406.18966)

- **Best practices and lessons learned on synthetic data for language models.** Liu, Ruibo, Wei, Jerry, Liu, Fangyu, Si, Chenglei, Zhang, Yanzhe, Rao, Jinmeng, Zheng, Steven, Peng, Daiyi, Yang, Diyi, Zhou, Denny, and others. *arXiv preprint arXiv:2404.07503* (2024) [[link]](https://openreview.net/pdf?id=OJaWBhh61C)

- **Self-Alignment with Instruction Backtranslation.** Li, Xian, Yu, Ping, Zhou, Chunting, Schick, Timo, Levy, Omer, Zettlemoyer, Luke, Weston, Jason E, and Lewis, Mike. *The Twelfth International Conference on Learning Representations* (2023) [[link]](https://arxiv.org/pdf/2308.06259)

- **Preference ranking optimization for human alignment.** Song, Feifan, Yu, Bowen, Li, Minghao, Yu, Haiyang, Huang, Fei, Li, Yongbin, and Wang, Houfeng. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29865/31509)

- **MathScale: Scaling Instruction Tuning for Mathematical Reasoning.** Tang, Zhengyang, Zhang, Xingxing, Wang, Benyou, and Wei, Furu. *Forty-first International Conference on Machine Learning* (No Year) [[link]](https://arxiv.org/pdf/2403.02884)

- **GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation.** Yoo, Kang Min, Park, Dongju, Kang, Jaewook, Lee, Sang-Woo, and Park, Woomyoung. *Findings of the Association for Computational Linguistics: EMNLP 2021* (2021) [[link]](https://arxiv.org/pdf/2104.08826)

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** Wang, Xuezhi, Wei, Jason, Schuurmans, Dale, Le, Quoc V, Chi, Ed H, Narang, Sharan, Chowdhery, Aakanksha, and Zhou, Denny. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2203.11171.pdf?trk=public_post_comment-text)

- **Tuning language models as training data generators for augmentation-enhanced few-shot learning.** Meng, Yu, Michalski, Martin, Huang, Jiaxin, Zhang, Yu, Abdelzaher, Tarek, and Han, Jiawei. *International Conference on Machine Learning* (2023) [[link]](https://proceedings.mlr.press/v202/meng23b/meng23b.pdf)

- **SASS: Self-Alignment with Semi-Supervised Instruction Data Generation.** Wang, Yue, Zhang, Haoke, Li, Juntao, Chang, Jinxiong, Zhang, Qishen, Liu, Zhongyi, Zhang, Guannan, and Zhang, Min. *No venue* (2023) [[link]](https://openreview.net/pdf?id=Q9vYgjcvrX)

- **Targen: Targeted data generation with large language models.** Gupta, Himanshu, Scaria, Kevin, Anantheswaran, Ujjwala, Verma, Shreyas, Parmar, Mihir, Sawant, Saurabh Arjun, Mishra, Swaroop, and Baral, Chitta. *arXiv preprint arXiv:2310.17876* (2023) [[link]](https://arxiv.org/pdf/2310.17876)

- **Let’s Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models.** Wang, Ruida, Zhou, Wangchunshu, and Sachan, Mrinmaya. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://arxiv.org/pdf/2310.13671)

- **Dail: Data augmentation for in-context learning via self-paraphrase.** Li, Dawei, Li, Yaxuan, Mekala, Dheeraj, Li, Shuyao, Wang, Xueqi, Hogan, William, Shang, Jingbo, and others. *arXiv preprint arXiv:2311.03319* (2023) [[link]](https://arxiv.org/pdf/2311.03319)

- **LongForm: Effective Instruction Tuning with Reverse Instructions.** K{\"o}ksal, Abdullatif, Schick, Timo, Korhonen, Anna, and Schuetze, Hinrich. *ICLR 2024 Workshop on Navigating and Addressing Data Problems for Foundation Models* (No Year) [[link]](https://arxiv.org/pdf/2304.08460)

- **Large language model as attributed training data generator: A tale of diversity and bias.** Yu, Yue, Zhuang, Yuchen, Zhang, Jieyu, Meng, Yu, Ratner, Alexander J, Krishna, Ranjay, Shen, Jiaming, and Zhang, Chao. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/ae9500c4f5607caf2eff033c67daa9d7-Paper-Datasets_and_Benchmarks.pdf)

- **Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing.** Xu, Zhangchen, Jiang, Fengqing, Niu, Luyao, Deng, Yuntian, Poovendran, Radha, Choi, Yejin, and Lin, Bill Yuchen. *arXiv preprint arXiv:2406.08464* (2024) [[link]](https://arxiv.org/pdf/2406.08464)

- **Scaling synthetic data creation with 1,000,000,000 personas.** Chan, Xin, Wang, Xiaoyang, Yu, Dian, Mi, Haitao, and Yu, Dong. *arXiv preprint arXiv:2406.20094* (2024) [[link]](https://arxiv.org/pdf/2406.20094?)

- **FANNO: Augmenting High-Quality Instruction Data with Open-Sourced LLMs Only.** Zhu, He, Su, Junyou, Lun, Tianle, Tao, Yicheng, Zhang, Wenjia, Fan, Zipei, and Chen, Guanhua. *arXiv preprint arXiv:2408.01323* (2024) [[link]](https://arxiv.org/pdf/2408.01323)

- **CorrSynth-A Correlated Sampling Method for Diverse Dataset Generation from LLMs.** Kowshik, Suhas, Divekar, Abhishek, and Malik, Vijit. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://arxiv.org/pdf/2411.08553)

- **SynthesizRR: Generating Diverse Datasets with Retrieval Augmentation.** Divekar, Abhishek, and Durrett, Greg. *arXiv preprint arXiv:2405.10040* (2024) [[link]](https://arxiv.org/pdf/2405.10040)

- **Optimizing Instruction Synthesis: Effective Exploration of Evolutionary Space with Tree Search.** Li, Chenglin, Chen, Qianglong, Li, Zhi, Tao, Feng, Li, Yicheng, Chen, Hao, Yu, Fei, and Zhang, Yin. *arXiv preprint arXiv:2410.10392* (2024) [[link]](https://arxiv.org/pdf/2410.10392?)

- **Assessing Empathy in Large Language Models with Real-World Physician-Patient Interactions.** Luo, Man, Warren, Christopher J, Cheng, Lu, Abdul-Muhsin, Haidar M, and Banerjee, Imon. *arXiv preprint arXiv:2405.16402* (2024) [[link]](https://arxiv.org/pdf/2405.16402)

- **Self-qa: Unsupervised knowledge guided language model alignment.** Zhang, Xuanyu, and Yang, Qing. *arXiv preprint arXiv:2305.11952* (2023) [[link]](https://arxiv.org/pdf/2305.11952)

- **Large Language Models Can Self-Improve.** Huang, Jiaxin, Gu, Shixiang, Hou, Le, Wu, Yuexin, Wang, Xuezhi, Yu, Hongkun, and Han, Jiawei. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2210.11610.pdf?src_trk=em6620554130ce70.4875613993116609)

- **Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning.** Yang, Zhaorui, Liu, Qian, Pang, Tianyu, Wang, Han, Feng, Haozhe, Zhu, Minfeng, and Chen, Wei. *arXiv preprint arXiv:2402.13669* (2024) [[link]](https://arxiv.org/pdf/2402.13669)

- **Self-Alignment of Large Language Models via Monopolylogue-based Social Scene Simulation.** Pang, Xianghe, Tang, Shuo, Ye, Rui, Xiong, Yuxin, Zhang, Bolun, Wang, Yanfeng, and Chen, Siheng. *arXiv preprint arXiv:2402.05699* (2024) [[link]](https://arxiv.org/pdf/2402.05699)

- **Mixture of insighTful Experts (MoTE): The Synergy of Thought Chains and Expert Mixtures in Self-Alignment.** Liu, Zhili, Gou, Yunhao, Chen, Kai, Hong, Lanqing, Gao, Jiahui, Mi, Fei, Zhang, Yu, Li, Zhenguo, Jiang, Xin, Liu, Qun, and others. *arXiv preprint arXiv:2405.00557* (2024) [[link]](https://arxiv.org/pdf/2405.00557)

- **Human-instruction-free llm self-alignment with limited samples.** Guo, Hongyi, Yao, Yuanshun, Shen, Wei, Wei, Jiaheng, Zhang, Xiaoying, Wang, Zhaoran, and Liu, Yang. *arXiv preprint arXiv:2401.06785* (2024) [[link]](https://arxiv.org/pdf/2401.06785)

- **Principle-driven self-alignment of language models from scratch with minimal human supervision.** Sun, Zhiqing, Shen, Yikang, Zhou, Qinhong, Zhang, Hongxin, Chen, Zhenfang, Cox, David, Yang, Yiming, and Gan, Chuang. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0764db1151b936aca59249e2c1386101-Paper-Conference.pdf)

- **Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping.** Wang, Haoyu, Ma, Guozheng, Meng, Ziqiao, Qin, Zeyu, Shen, Li, Zhang, Zhong, Wu, Bingzhe, Liu, Liu, Bian, Yatao, Xu, Tingyang, and others. *arXiv preprint arXiv:2402.07610* (2024) [[link]](https://arxiv.org/pdf/2402.07610)

- **Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources.** Lupidi, Alisia, Gemmell, Carlos, Cancedda, Nicola, Dwivedi-Yu, Jane, Weston, Jason, Foerster, Jakob, Raileanu, Roberta, and Lomeli, Maria. *arXiv preprint arXiv:2409.08239* (2024) [[link]](https://arxiv.org/pdf/2409.08239)

- **ControlMath: Controllable Data Generation Promotes Math Generalist Models.** Chen, Nuo, Wu, Ning, Chang, Jianhui, Shou, Linjun, and Li, Jia. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://arxiv.org/pdf/2409.15376)


### Label

- **The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators** Tzu-Heng Huang, Catherine Cao, Vaishnavi Bhargava, Frederic Sala. *Advances in Neural Information Processing Systems (2024, Spotlight)* [[link]](https://arxiv.org/abs/2407.11004) 

- **AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving.** Liang, Mingfu, Su, Jong-Chyi, Schulter, Samuel, Garg, Sparsh, Zhao, Shiyu, Wu, Ying, Chandraker, Manmohan. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2024) [[link]](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.html)

- **Towards Automating Text Annotation: A Case Study on Semantic Proximity Annotation using GPT-4.** Yadav, Sachin, Choppa, Tejaswi, and Schlechtweg, Dominik. *arXiv preprint arXiv:2407.04130* (2024) [[link]](https://arxiv.org/pdf/2407.04130)

- **Is a Large Language Model a Good Annotator for Event Extraction?.** Chen, Ruirui, Qin, Chengwei, Jiang, Weifeng, and Choi, Dongkyu. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29730/31254)

- **Zero-Shot Topic Classification of Column Headers: Leveraging LLMs for Metadata Enrichment.** Martorana, Margherita, Kuhn, Tobias, Stork, Lise, and van Ossenbruggen, Jacco. *Knowledge Graphs in the Age of Language Models and Neuro-Symbolic AI* (2024) [[link]](https://ebooks.iospress.nl/pdf/doi/10.3233/SSW240006)

- **Enhancing Text Annotation through Rationale-Driven Collaborative Few-Shot Prompting.** Wu, Jianfei, Wang, Xubin, and Jia, Weijia. *arXiv preprint arXiv:2409.09615* (2024) [[link]](https://arxiv.org/pdf/2409.09615)

- **Can LLMs Replace Manual Annotation of Software Engineering Artifacts?.** Ahmed, Toufique, Devanbu, Premkumar, Treude, Christoph, and Pradel, Michael. *arXiv preprint arXiv:2408.05534* (2024) [[link]](https://arxiv.org/pdf/2408.05534)

- **CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation.** Li, Minzhi, Shi, Taiwei, Ziems, Caleb, Kan, Min-Yen, Chen, Nancy, Liu, Zhengyuan, and Yang, Diyi. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2310.15638)

- **Leveraging Large Language Models and Weak Supervision for Social Media data annotation: an evaluation using COVID-19 self-reported vaccination tweets.** Tekumalla, Ramya, and Banda, Juan M. *International Conference on Human-Computer Interaction* (2023) [[link]](https://arxiv.org/pdf/2309.06503)

- **Best Practices for Text Annotation with Large Language Models.** T{\"o}rnberg, Petter. *arXiv preprint arXiv:2402.05129* (2024) [[link]](https://arxiv.org/pdf/2402.05129)

- **Can large language models fix data annotation errors? an empirical study using debatepedia for query-focused text summarization.** Laskar, Md Tahmid Rahman, Rahman, Mizanur, Jahan, Israt, Hoque, Enamul, and Huang, Jimmy. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://aclanthology.org/2023.findings-emnlp.686.pdf)

- **Large language models improve annotation of prokaryotic viral proteins.** Flamholz, Zachary N, Biller, Steven J, and Kelly, Libusha. *Nature Microbiology* (2024) [[link]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11311208/pdf/nihms-2009389.pdf)

- **Prompting-based Synthetic Data Generation for Few-Shot Question Answering.** Schmidt, Maximilian, Bartezzaghi, Andrea, and Vu, Ngoc Thang. *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (2024) [[link]](https://arxiv.org/pdf/2405.09335)

- **UniGen: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation.** Choi, Juhwan, Kim, Yeonghwa, Yu, Seunguk, Yun, JungMin, and Kim, YoungBin. *arXiv preprint arXiv:2405.01022* (2024) [[link]](https://arxiv.org/pdf/2405.01022)

- **Optimizing Code Retrieval: High-Quality and Scalable Dataset Annotation through Large Language Models.** Li, Rui, Liu, Qi, He, Liyang, Zhang, Zheng, Zhang, Hao, Ye, Shengyu, Lu, Junyu, and Huang, Zhenya. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://aclanthology.org/2024.emnlp-main.123.pdf)

- **Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation.** Choi, Juhwan, Yun, Jungmin, Jin, Kyohoon, and Kim, YoungBin. *arXiv preprint arXiv:2404.09682* (2024) [[link]](https://arxiv.org/pdf/2404.09682)

- **Fill In The Gaps: Model Calibration and Generalization with Synthetic Data.** Ba, Yang, Mancenido, Michelle, and Pan, Rong. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://arxiv.org/pdf/2410.10864)

- **Are Expert-Level Language Models Expert-Level Annotators?.** Yu-Min Tseng, Wei-Lin Chen, Chung-Chi Chen and Hsin-Hsi Chen *arXiv preprint arXiv: 2410.03254* (2024) [[link]](https://arxiv.org/abs/2410.03254)



### Rationale

- **Large language models are zero-shot reasoners.** Kojima, Takeshi, Gu, Shixiang Shane, Reid, Machel, Matsuo, Yutaka, and Iwasawa, Yusuke. *Advances in neural information processing systems* (2022) [[link]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

- **Reasoning with Language Model is Planning with World Model.** Hao, Shibo, Gu, Yi, Ma, Haodi, Hong, Joshua, Wang, Zhen, Wang, Daisy, and Hu, Zhiting. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.14992)

- **Tree of thoughts: Deliberate problem solving with large language models.** Yao, Shunyu, Yu, Dian, Zhao, Jeffrey, Shafran, Izhak, Griffiths, Tom, Cao, Yuan, and Narasimhan, Karthik. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf)

- **Graph of thoughts: Solving elaborate problems with large language models.** Besta, Maciej, Blach, Nils, Kubicek, Ales, Gerstenberger, Robert, Podstawski, Michal, Gianinazzi, Lukas, Gajda, Joanna, Lehmann, Tomasz, Niewiadomski, Hubert, Nyczyk, Piotr, and others. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29720/31236)

- **Beyond chain-of-thought, effective graph-of-thought reasoning in large language models.** Yao, Yao, Li, Zuchao, and Zhao, Hai. *arXiv preprint arXiv:2305.16582* (2023) [[link]](https://arxiv.org/pdf/2305.16582)

- **Chain-of-table: Evolving tables in the reasoning chain for table understanding.** Wang, Zilong, Zhang, Hao, Li, Chun-Liang, Eisenschlos, Julian Martin, Perot, Vincent, Wang, Zifeng, Miculicich, Lesly, Fujii, Yasuhisa, Shang, Jingbo, Lee, Chen-Yu, and others. *arXiv preprint arXiv:2401.04398* (2024) [[link]](https://arxiv.org/pdf/2401.04398)

- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.** Chen, Wenhu, Ma, Xueguang, Wang, Xinyi, and Cohen, William W. *Transactions on Machine Learning Research* (2023) [[link]](https://arxiv.org/pdf/2211.12588)

- **The art of SOCRATIC QUESTIONING: Recursive thinking with large language models.** Qi, Jingyuan, Xu, Zhiyang, Shen, Ying, Liu, Minqian, Jin, Di, Wang, Qifan, and Huang, Lifu. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.14999)

- **Interpreting Pretrained Language Models via Concept Bottlenecks.** Tan, Zhen, Cheng, Lu, Wang, Song, Bo, Yuan, Li, Jundong, and Liu, Huan. *arXiv preprint arXiv:2311.05014* (2023) [[link]](https://arxiv.org/pdf/2311.05014)

- **PINTO: Faithful Language Reasoning Using Prompt-Generated Rationales.** Wang, PeiFeng, Chan, Aaron, Ilievski, Filip, Chen, Muhao, and Ren, Xiang. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2211.01562)

- **LogiCoT: Logical Chain-of-Thought Instruction Tuning.** Liu, Hanmeng, Teng, Zhiyang, Cui, Leyang, Zhang, Chaoli, Zhou, Qiji, and Zhang, Yue. *The 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.12147)

- **Distilling Reasoning Capabilities into Smaller Language Models.** Shridhar, Kumar, Stolfo, Alessandro, and Sachan, Mrinmaya. *Findings of the Association for Computational Linguistics: ACL 2023* (2023) [[link]](https://arxiv.org/pdf/2212.00193)

- **Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks.** Kang, Minki, Lee, Seanie, Baek, Jinheon, Kawaguchi, Kenji, and Hwang, Sung Ju. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/97faedc90260eae5c400f92d5831c3d7-Paper-Conference.pdf)

- **Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data.** Zhou, Jiaming, Ghaddar, Abbas, Zhang, Ge, Ma, Liheng, Hu, Yaochen, Pal, Soumyasundar, Coates, Mark, Wang, Bin, Zhang, Yingxue, and Hao, Jianye. *arXiv preprint arXiv:2409.12437* (2024) [[link]](https://arxiv.org/pdf/2409.12437)

- **Making Pre-trained Language Models Better Few-shot Learners.** Gao, Tianyu, Fisch, Adam, and Chen, Danqi. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)* (2021) [[link]](https://aclanthology.org/2021.acl-long.295.pdf)

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** Wang, Xuezhi, Wei, Jason, Schuurmans, Dale, Le, Quoc V, Chi, Ed H, Narang, Sharan, Chowdhery, Aakanksha, and Zhou, Denny. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2203.11171.pdf?trk=public_post_comment-text)

- **Universal self-consistency for large language model generation.** Chen, Xinyun, Aksitov, Renat, Alon, Uri, Ren, Jie, Xiao, Kefan, Yin, Pengcheng, Prakash, Sushant, Sutton, Charles, Wang, Xuezhi, and Zhou, Denny. *arXiv preprint arXiv:2311.17311* (2023) [[link]](https://arxiv.org/pdf/2311.17311)

- **Plan, Verify and Switch: Integrated Reasoning with Diverse X-of-Thoughts.** Liu, Tengxiao, Guo, Qipeng, Yang, Yuqing, Hu, Xiangkun, Zhang, Yue, Qiu, Xipeng, and Zhang, Zheng. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2310.14628)

- **Eliminating Reasoning via Inferring with Planning: A New Framework to Guide LLMs' Non-linear Thinking.** Tong, Yongqi, Wang, Yifan, Li, Dawei, Wang, Sizhe, Lin, Zi, Han, Simeng, and Shang, Jingbo. *arXiv preprint arXiv:2310.12342* (2023) [[link]](https://arxiv.org/pdf/2310.12342.pdf?fbclid=IwAR10rCAqJZCMGgMTuUZnUOqyNbpZ8rkF6b29Smdtlbjs8gTUQ3309vytK28)

- **It's Not Easy Being Wrong: Evaluating Process of Elimination Reasoning in Large Language Models.** Balepur, Nishant, Palta, Shramay, and Rudinger, Rachel. *arXiv preprint arXiv:2311.07532* (2023) [[link]](https://arxiv.org/pdf/2311.07532)

- **POE: Process of Elimination for Multiple Choice Reasoning.** Ma, Chenkai, and Du, Xinya. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2310.15575)

- **Exchange-of-thought: Enhancing large language model capabilities through cross-model communication.** Yin, Zhangyue, Sun, Qiushi, Chang, Cheng, Guo, Qipeng, Dai, Junqi, Huang, Xuan-Jing, and Qiu, Xipeng. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2406.08979)

- **Encouraging divergent thinking in large language models through multi-agent debate.** Liang, Tian, He, Zhiwei, Jiao, Wenxiang, Wang, Xing, Wang, Yan, Wang, Rui, Yang, Yujiu, Tu, Zhaopeng, and Shi, Shuming. *arXiv preprint arXiv:2305.19118* (2023) [[link]](https://arxiv.org/pdf/2305.19118)

- **Towards reasoning in large language models via multi-agent peer review collaboration.** Xu, Zhenran, Shi, Senbao, Hu, Baotian, Yu, Jindi, Li, Dongfang, Zhang, Min, and Wu, Yuxiang. *arXiv preprint arXiv:2311.08152* (2023) [[link]](https://arxiv.org/pdf/2311.08152)

- **Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization.** Liu, Zijun, Zhang, Yanzhe, Li, Peng, Liu, Yang, and Yang, Diyi. *arXiv preprint arXiv:2310.02170* (2023) [[link]](https://arxiv.org/pdf/2310.02170)

- **Large Language Models Can Learn Temporal Reasoning.** Siheng Xiong, Ali Payani, Ramana Kompella and Faramarz Fekri *arXiv preprint arXiv: 2401.06853* (2024) [[link]](https://arxiv.org/pdf/2401.06853)

- **Large Language Models Can Self-Improve.** Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu and Jiawei Han *arXiv preprint arXiv: 2210.11610* (2022) [[link]](https://arxiv.org/pdf/2210.11610)

- **Case2Code: Learning Inductive Reasoning with Synthetic Data.** Yunfan Shao, Linyang Li, Yichuan Ma, Peiji Li, Demin Song, Qinyuan Cheng, Shimin Li, Xiaonan Li, Pengyu Wang, Qipeng Guo, Hang Yan, Xipeng Qiu, Xuanjing Huang and Dahua Lin *arXiv preprint arXiv: 2407.12504* (2024) [[link]](https://arxiv.org/pdf/2407.12504)

- **Can LLMs Reason in the Wild with Programs?** Yuan Yang, Siheng Xiong, Ali Payani, Ehsan Shareghi and Faramarz Fekri *arXiv preprint arXiv: 2406.13764* (2024) [[link]](https://arxiv.org/pdf/2406.13764)

- **Advancing LLM Reasoning Generalists with Preference Trees.** Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu and Maosong Sun *arXiv preprint arXiv: 2406.13764* (2024) [[link]](https://arxiv.org/pdf/2404.02078)

- **T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Large Language Model Signals for Science Question Answering.** Lei Wang, Yi Hu, Jiabang He, Xing Xu, Ning Liu, Hui Liu and Heng Tao Shen *AAAI Conference on Artificial Intelligence (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29884)

- **Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model.** Siheng Xiong, Ali Payani, Yuan Yang and Faramarz Fekri *arXiv preprint arXiv: 2410.03136* (2024) [[link]](https://arxiv.org/pdf/2410.03136)

- **LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models.** Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra and Chitta Baral *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (2024) [[link]](https://aclanthology.org/2024.acl-long.739.pdf)

- **LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning.** Jin Jiang, Yuchen Yan, Yang Liu, Yonggang Jin, Shuai Peng, Mengdi Zhang, Xunliang Cai, Yixin Cao, Liangcai Gao and Zhi Tang *arXiv preprint arXiv: 2409.12929* (2024) [[link]](https://arxiv.org/pdf/2409.12929)

- **Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning.** Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan and Baocai Yin *arXiv preprint arXiv: 2405.14170* (2024) [[link]](https://arxiv.org/pdf/2405.14170)

- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4.** Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi and Ahmed Awadallah *arXiv preprint arXiv: 2306.02707* (2023) [[link]](https://arxiv.org/pdf/2306.02707)

- **Orca 2: Teaching Small Language Models How to Reason.** Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour and Ahmed Awadallah *arXiv preprint arXiv: 2311.11045* (2023) [[link]](https://arxiv.org/pdf/2311.11045)

- **ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement.** Xiangyu Peng, Congying Xia, Xinyi Yang, Caiming Xiong, Chien-Sheng Wu and Chen Xing *arXiv preprint arXiv: 2410.02108* (2024) [[link]](https://arxiv.org/pdf/2410.02108)

- **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman Garg, Virginia Smith and Aviral Kumar *arXiv preprint arXiv: 2406.14532* (2024) [[link]](https://arxiv.org/pdf/2406.14532)

- **STaR: Bootstrapping Reasoning With Reasoning.** Eric Zelikman, Yuhuai Wu, Jesse Mu and Noah D. Goodman *arXiv preprint arXiv: 2203.14465* (2022) [[link]](https://arxiv.org/pdf/2203.14465)

- **Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning.** Bahare Fatemi, Mehran Kazemi, Anton Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palowitch, Sungyong Seo, Jonathan Halcrow and Bryan Perozzi *arXiv preprint arXiv: 2406.09170* (2024) [[link]](https://arxiv.org/pdf/2406.09170)

- **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing.** Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi and Dong Yu *arXiv preprint arXiv: 2404.12253* (2024) [[link]](https://arxiv.org/pdf/2404.12253)

- **Understanding Social Reasoning in Language Models with Language Models.** Kanishk Gandhi, Jan-Philipp Fraenken, Tobias Gerstenberg and Noah Goodman Advances in Neural Information Processing Systems (2023) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2b9efb085d3829a2aadffab63ba206de-Paper-Datasets_and_Benchmarks.pdf)

- **INSTRUCTRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales.** Zhepei Wei, Wei-Lin Chen and Yu Meng *arXiv preprint arXiv: 2406.13629* (2024) [[link]](https://arxiv.org/pdf/2406.13629)

<!-- - **[TITLE] .** [AUTHOR] *arXiv preprint arXiv: [ID]* () [[link]]() -->

### Pairwise Feedback

- **Constitutional ai: Harmlessness from ai feedback.** Bai, Yuntao, Kadavath, Saurav, Kundu, Sandipan, Askell, Amanda, Kernion, Jackson, Jones, Andy, Chen, Anna, Goldie, Anna, Mirhoseini, Azalia, McKinnon, Cameron, and others. *arXiv preprint arXiv:2212.08073* (2022) [[link]](https://arxiv.org/pdf/2212.08073.pdf?trk=public_post_comment-text)

- **Rlaif: Scaling reinforcement learning from human feedback with ai feedback.** Lee, Harrison, Phatale, Samrat, Mansoor, Hassan, Lu, Kellie, Mesnard, Thomas, Bishop, Colton, Carbune, Victor, and Rastogi, Abhinav. *arXiv preprint arXiv:2309.00267* (2023) [[link]](https://openreview.net/pdf?id=AAxIs3D2ZZ)

- **Self-rewarding language models.** Yuan, Weizhe, Pang, Richard Yuanzhe, Cho, Kyunghyun, Sukhbaatar, Sainbayar, Xu, Jing, and Weston, Jason. *arXiv preprint arXiv:2401.10020* (2024) [[link]](http://readwise-assets.s3.amazonaws.com/media/wisereads/articles/self-rewarding-language-models/2401.10020.pdf)

- **SALMON: Self-Alignment with Instructable Reward Models.** Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David D. Cox, Yiming Yang, and Chuang Gan. *No venue* (2023) [[link]](https://openreview.net/pdf?id=xJbsmB8UMx)

- **Principle-driven self-alignment of language models from scratch with minimal human supervision.** Sun, Zhiqing, Shen, Yikang, Zhou, Qinhong, Zhang, Hongxin, Chen, Zhenfang, Cox, David, Yang, Yiming, and Gan, Chuang. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/0764db1151b936aca59249e2c1386101-Paper-Conference.pdf)

- **Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation.** Zhang, Xiaoying, Peng, Baolin, Tian, Ye, Zhou, Jingyan, Jin, Lifeng, Song, Linfeng, Mi, Haitao, and Meng, Helen. *arXiv preprint arXiv:2402.09267* (2024) [[link]](https://arxiv.org/pdf/2402.09267)

- **West-of-N: Synthetic Preference Generation for Improved Reward Modeling.** Pace, Aliz{\'e}e, Mallinson, Jonathan, Malmi, Eric, Krause, Sebastian, and Severyn, Aliaksei. *arXiv preprint arXiv:2401.12086* (2024) [[link]](https://arxiv.org/pdf/2401.12086)

- **Learning Reward for Robot Skills Using Large Language Models via Self-Alignment.** Zeng, Yuwei, Mu, Yao, and Shao, Lin. *arXiv preprint arXiv:2405.07162* (2024) [[link]](https://arxiv.org/pdf/2405.07162)

- **Improving Language Model Reasoning with Self-motivated Learning.** Feng, Yunlong, Xu, Yang, Qin, Libo, Wang, Yasheng, and Che, Wanxiang. *arXiv preprint arXiv:2404.07017* (2024) [[link]](https://arxiv.org/pdf/2404.07017)

- **Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection.** Lee, Kyungjae, Hwang, Dasol, Park, Sunghyun, Jang, Youngsoo, and Lee, Moontae. *arXiv preprint arXiv:2403.14238* (2024) [[link]](https://arxiv.org/pdf/2403.14238)

- **Aligning Large Language Models through Synthetic Feedback.** Kim, Sungdong, Bae, Sanghwan, Shin, Jamin, Kang, Soyoung, Kwak, Donghyun, Yoo, Kang, and Seo, Minjoon. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.13735)

- **Optimizing Language Model's Reasoning Abilities with Weak Supervision.** Tong, Yongqi, Wang, Sizhe, Li, Dawei, Wang, Yifan, Han, Simeng, Lin, Zi, Huang, Chengsong, Huang, Jiaxin, and Shang, Jingbo. *arXiv preprint arXiv:2405.04086* (2024) [[link]](https://arxiv.org/pdf/2405.04086)

- **RLCD: Reinforcement learning from contrastive distillation for LM alignment.** Yang, Kevin, Klein, Dan, Celikyilmaz, Asli, Peng, Nanyun, and Tian, Yuandong. *The Twelfth International Conference on Learning Representations* (2023) [[link]](https://arxiv.org/pdf/2307.12950.pdf?curius=4200)

- **Contrastive post-training large language models on data curriculum.** Xu, Canwen, Rosset, Corby, Del Corro, Luciano, Mahajan, Shweti, McAuley, Julian, Neville, Jennifer, Awadallah, Ahmed Hassan, and Rao, Nikhil. *arXiv preprint arXiv:2310.02263* (2023) [[link]](https://arxiv.org/pdf/2310.02263)



### Textual Feedback

- **Automatically correcting large language models: Surveying the landscape of diverse automated correction strategies.** Pan, Liangming, Saxon, Michael, Xu, Wenda, Nathani, Deepak, Wang, Xinyi, and Wang, William Yang. *Transactions of the Association for Computational Linguistics* (2024) [[link]](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00660/2369509/tacl_a_00660.pdf)

- **Self-refine: Iterative refinement with self-feedback.** Madaan, Aman, Tandon, Niket, Gupta, Prakhar, Hallinan, Skyler, Gao, Luyu, Wiegreffe, Sarah, Alon, Uri, Dziri, Nouha, Prabhumoye, Shrimai, Yang, Yiming, and others. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf)

- **Reflexion: Language agents with verbal reinforcement learning.** Shinn, Noah, Cassano, Federico, Gopinath, Ashwin, Narasimhan, Karthik, and Yao, Shunyu. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)

- **Do as i can, not as i say: Grounding language in robotic affordances.** Brohan, Anthony, Chebotar, Yevgen, Finn, Chelsea, Hausman, Karol, Herzog, Alexander, Ho, Daniel, Ibarz, Julian, Irpan, Alex, Jang, Eric, Julian, Ryan, and others. *Conference on robot learning* (2023) [[link]](https://proceedings.mlr.press/v205/ichter23a.html)

- **Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment.** Ning, Kun-Peng, Yang, Shuo, Liu, Yu-Yang, Yao, Jia-Yu, Liu, Zhen-Hui, Wang, Yu, Pang, Ming, and Yuan, Li. *arXiv preprint arXiv:2402.01830* (2024) [[link]](https://arxiv.org/pdf/2402.01830)

- **A New Benchmark and Reverse Validation Method for Passage-level Hallucination Detection.** Yang, Shiping, Sun, Renliang, and Wan, Xiaojun. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://arxiv.org/pdf/2310.06498)

- **Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models.** Manakul, Potsawee, Liusie, Adian, and Gales, Mark JF. *arXiv preprint arXiv:2303.08896* (2023) [[link]](https://arxiv.org/pdf/2303.08896)

- **Improving factuality and reasoning in language models through multiagent debate.** Du, Yilun, Li, Shuang, Torralba, Antonio, Tenenbaum, Joshua B, and Mordatch, Igor. *arXiv preprint arXiv:2305.14325* (2023) [[link]](https://arxiv.org/pdf/2305.14325.pdf?trk=article-ssr-frontend-pulse_x-social-details_comments-action_comment-text)

- **Towards reasoning in large language models via multi-agent peer review collaboration.** Xu, Zhenran, Shi, Senbao, Hu, Baotian, Yu, Jindi, Li, Dongfang, Zhang, Min, and Wu, Yuxiang. *arXiv preprint arXiv:2311.08152* (2023) [[link]](https://arxiv.org/pdf/2311.08152)

- **Lm vs lm: Detecting factual errors via cross examination.** Cohen, Roi, Hamri, May, Geva, Mor, and Globerson, Amir. *arXiv preprint arXiv:2305.13281* (2023) [[link]](https://arxiv.org/pdf/2305.13281)

- **Prd: Peer rank and discussion improve large language model based evaluations.** Li, Ruosen, Patel, Teerth, and Du, Xinya. *arXiv preprint arXiv:2307.02762* (2023) [[link]](https://arxiv.org/pdf/2307.02762)

- **PRE: A Peer Review Based Large Language Model Evaluator.** Chu, Zhumin, Ai, Qingyao, Tu, Yiteng, Li, Haitao, and Liu, Yiqun. *arXiv preprint arXiv:2401.15641* (2024) [[link]](https://arxiv.org/pdf/2401.15641)

- **Learning from mistakes via cooperative study assistant for large language models.** Wang, Danqing, and Li, Lei. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.13829)

- **Learning from mistakes makes llm better reasoner.** An, Shengnan, Ma, Zexiong, Lin, Zeqi, Zheng, Nanning, Lou, Jian-Guang, and Chen, Weizhu. *arXiv preprint arXiv:2310.20689* (2023) [[link]](https://arxiv.org/pdf/2310.20689.pdf?trk=public_post_comment-text)

- **Gaining wisdom from setbacks: Aligning large language models via mistake analysis.** Chen, Kai, Wang, Chunwei, Yang, Kuo, Han, Jianhua, Hong, Lanqing, Mi, Fei, Xu, Hang, Liu, Zhengying, Huang, Wenyong, Li, Zhenguo, and others. *arXiv preprint arXiv:2310.10477* (2023) [[link]](https://arxiv.org/pdf/2310.10477)

- **Can LLMs Learn from Previous Mistakes? Investigating LLMs' Errors to Boost for Reasoning.** Tong, Yongqi, Li, Dawei, Wang, Sizhe, Wang, Yujia, Teng, Fei, and Shang, Jingbo. *arXiv preprint arXiv:2403.20046* (2024) [[link]](https://arxiv.org/pdf/2403.20046)

### Other Domain-Specific Data

- **AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving.** Liang, Mingfu, Su, Jong-Chyi, Schulter, Samuel, Garg, Sparsh, Zhao, Shiyu, Wu, Ying, Chandraker, Manmohan. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2024) [[link]](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.html)

- **SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization.** Kim, Hyunwoo, Hessel, Jack, Jiang, Liwei, West, Peter, Lu, Ximing, Yu, Youngjae, Zhou, Pei, Bras, Ronan, Alikhani, Malihe, Kim, Gunhee, and others. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2212.10465)

- **Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data.** Xu, Canwen, Guo, Daya, Duan, Nan, and McAuley, Julian. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2304.01196)

- **PLACES: Prompting Language Models for Social Conversation Synthesis.** Chen, Maximillian, Papangelis, Alexandros, Tao, Chenyang, Kim, Seokhwan, Rosenbaum, Andy, Liu, Yang, Yu, Zhou, and Hakkani-Tur, Dilek. *Findings of the Association for Computational Linguistics: EACL 2023* (2023) [[link]](https://arxiv.org/pdf/2302.03269)

- **Camel: Communicative agents for" mind" exploration of large language model society.** Li, Guohao, Hammoud, Hasan, Itani, Hani, Khizbullin, Dmitrii, and Ghanem, Bernard. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/a3621ee907def47c1b952ade25c67698-Paper-Conference.pdf)

- **CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models.** Wang, Song, Wang, Peng, Zhou, Tong, Dong, Yushun, Tan, Zhen, and Li, Jundong. *arXiv preprint arXiv:2407.02408* (2024) [[link]](https://arxiv.org/pdf/2407.02408)

- **Synth-Empathy: Towards High-Quality Synthetic Empathy Data.** Liang, Hao, Sun, Linzhuang, Wei, Jingxuan, Huang, Xijie, Sun, Linkun, Yu, Bihui, He, Conghui, and Zhang, Wentao. *arXiv preprint arXiv:2407.21669* (2024) [[link]](https://arxiv.org/pdf/2407.21669)

- **AugESC: Dialogue Augmentation with Large Language Models for Emotional Support Conversation.** Zheng, Chujie, Sabour, Sahand, Wen, Jiaxin, Zhang, Zheng, and Huang, Minlie. *Findings of the Association for Computational Linguistics: ACL 2023* (2023) [[link]](https://arxiv.org/pdf/2202.13047)

- **Weakly Supervised Data Augmentation Through Prompting for Dialogue Understanding.** Chen, Maximillian, Papangelis, Alexandros, Tao, Chenyang, Rosenbaum, Andy, Kim, Seokhwan, Liu, Yang, Yu, Zhou, and Hakkani-Tur, Dilek. *NeurIPS 2022 Workshop on Synthetic Data for Empowering ML Research* (2022) [[link]](https://arxiv.org/pdf/2210.14169)

- **Reflect, Not Reflex: Inference-Based Common Ground Improves Dialogue Response Quality.** Zhou, Pei, Cho, Hyundong, Jandaghi, Pegah, Lee, Dong-Ho, Lin, Bill Yuchen, Pujara, Jay, and Ren, Xiang. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (2022) [[link]](https://arxiv.org/pdf/2211.09267)

- **Fostering Natural Conversation in Large Language Models with NICO: a Natural Interactive COnversation dataset.** Renliang Sun, Mengyuan Liu, Shiping Yang, Rui Wang, Junqing He, and Jiaxing Zhang. *ArXiv* (2024) [[link]](https://arxiv.org/pdf/2408.09330)

- **ASDOT: Any-Shot Data-to-Text Generation with Pretrained Language Models.** Xiang, Jiannan, Liu, Zhengzhong, Zhou, Yucheng, Xing, Eric, and Hu, Zhiting. *Findings of the Association for Computational Linguistics: EMNLP 2022* (2022) [[link]](https://arxiv.org/pdf/2210.04325)

- **Contextualization distillation from large language model for knowledge graph completion.** Li, Dawei, Tan, Zhen, Chen, Tianlong, and Liu, Huan. *arXiv preprint arXiv:2402.01729* (2024) [[link]](https://arxiv.org/pdf/2402.01729)

- **Towards Ontology-Enhanced Representation Learning for Large Language Models.** Ronzano, Francesco, and Nanavati, Jay. *arXiv preprint arXiv:2405.20527* (2024) [[link]](https://arxiv.org/pdf/2405.20527)

- **TILP}: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs.** Siheng Xiong, Yuan Yang, Faramarz Fekri, and James Clayton Kerce. *The Eleventh International Conference on Learning Representations * (2023) [[link]](https://arxiv.org/pdf/2402.12309)

- **Teilp: Time prediction over knowledge graphs via logical reasoning.** Xiong, Siheng, Yang, Yuan, Payani, Ali, Kerce, James C, and Fekri, Faramarz. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29544/30907)

- **Codekgc: Code language model for generative knowledge graph construction.** Bi, Zhen, Chen, Jing, Jiang, Yinuo, Xiong, Feiyu, Guo, Wei, Chen, Huajun, and Zhang, Ningyu. *ACM Transactions on Asian and Low-Resource Language Information Processing* (2024) [[link]](https://arxiv.org/pdf/2304.09048)

- **DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature.** Li, Dawei, Yang, Shu, Tan, Zhen, Baik, Jae Young, Yun, Sunkwon, Lee, Joseph, Chacko, Aaron, Hou, Bojian, Duong-Tran, Duy, Ding, Ying, and others. *arXiv preprint arXiv:2405.04819* (2024) [[link]](https://arxiv.org/pdf/2409.13731)

- **Automated Construction of Theme-specific Knowledge Graphs.** Ding, Linyi, Zhou, Sizhe, Xiao, Jinfeng, and Han, Jiawei. *arXiv preprint arXiv:2404.19146* (2024) [[link]](https://arxiv.org/pdf/2404.19146)

- **Large Language Models Can Learn Temporal Reasoning.** Siheng Xiong, Ali Payani, Ramana Kompella and Faramarz Fekri *arXiv preprint arXiv: 2401.06853* (2024) [[link]](https://arxiv.org/pdf/2401.06853)

- **Moving from Tabular Knowledge Graph Quality Assessment to RDF Triples Leveraging ChatGPT.** Tuozzo, Gabriele. *No venue* (2022) [[link]](https://dqmlkg.github.io/assets/paper_3.pdf)

- **Language models as zero-shot planners: Extracting actionable knowledge for embodied agents.** Huang, Wenlong, Abbeel, Pieter, Pathak, Deepak, and Mordatch, Igor. *International Conference on Machine Learning* (2022) [[link]](https://proceedings.mlr.press/v162/huang22a/huang22a.pdf)

- **Do as i can, not as i say: Grounding language in robotic affordances.** Brohan, Anthony, Chebotar, Yevgen, Finn, Chelsea, Hausman, Karol, Herzog, Alexander, Ho, Daniel, Ibarz, Julian, Irpan, Alex, Jang, Eric, Julian, Ryan, and others. *Conference on robot learning* (2023) [[link]](https://proceedings.mlr.press/v205/ichter23a.html)

- **Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning.** Rana, Krishan, Haviland, Jesse, Garg, Sourav, Abou-Chakra, Jad, Reid, Ian, and Suenderhauf, Niko. *7th Annual Conference on Robot Learning* (2023) [[link]](https://openreview.net/pdf?id=wMpOMO0Ss7a)

- **Progprompt: Generating situated robot task plans using large language models.** Singh, Ishika, Blukis, Valts, Mousavian, Arsalan, Goyal, Ankit, Xu, Danfei, Tremblay, Jonathan, Fox, Dieter, Thomason, Jesse, and Garg, Animesh. *2023 IEEE International Conference on Robotics and Automation (ICRA)* (2023) [[link]](https://arxiv.org/pdf/2209.11302)

- **Text2motion: From natural language instructions to feasible plans.** Lin, Kevin, Agia, Christopher, Migimatsu, Toki, Pavone, Marco, and Bohg, Jeannette. *Autonomous Robots* (2023) [[link]](https://arxiv.org/pdf/2303.12153)

- **GenSim: Generating Robotic Simulation Tasks via Large Language Models.** Wang, Lirui, Ling, Yiyang, Yuan, Zhecheng, Shridhar, Mohit, Bao, Chen, Qin, Yuzhe, Wang, Bailin, Xu, Huazhe, and Wang, Xiaolong. *The Twelfth International Conference on Learning Representations* (2023) [[link]](https://arxiv.org/pdf/2310.01361)

- **Scaling up and distilling down: Language-guided robot skill acquisition.** Ha, Huy, Florence, Pete, and Song, Shuran. *Conference on Robot Learning* (2023) [[link]](https://proceedings.mlr.press/v229/ha23a/ha23a.pdf)

- **Reward Design with Language Models.** Kwon, Minae, Xie, Sang Michael, Bullard, Kalesha, and Sadigh, Dorsa. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2303.00001)

- **Guiding pretraining in reinforcement learning with large language models.** Du, Yuqing, Watkins, Olivia, Wang, Zihan, Colas, C{\'e}dric, Darrell, Trevor, Abbeel, Pieter, Gupta, Abhishek, and Andreas, Jacob. *International Conference on Machine Learning* (2023) [[link]](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

- **Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data.** Li, Yanda, Zhang, Chi, Yu, Gang, Wang, Zhibin, Fu, Bin, Lin, Guosheng, Shen, Chunhua, Chen, Ling, and Wei, Yunchao. *arXiv preprint arXiv:2308.10253* (2023) [[link]](https://arxiv.org/pdf/2308.10253)

- **Lamm: Language-assisted multi-modal instruction-tuning dataset, framework, and benchmark.** Yin, Zhenfei, Wang, Jiong, Cao, Jianjian, Shi, Zhelun, Liu, Dingning, Li, Mukai, Huang, Xiaoshui, Wang, Zhiyong, Sheng, Lu, Bai, Lei, and others. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/548a41b9cac6f50dccf7e63e9e1b1b9b-Paper-Datasets_and_Benchmarks.pdf)

- **TOMGPT: Reliable Text-Only Training Approach for Cost-Effective Multi-modal Large Language Model.** Chen, Yunkai, Wang, Qimeng, Wu, Shiwei, Gao, Yan, Xu, Tong, and Hu, Yao. *ACM Transactions on Knowledge Discovery from Data* (2024) [[link]](https://dl.acm.org/doi/abs/10.1145/3654674)

- **MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct.** Luo, Run, Zhang, Haonan, Chen, Longze, Lin, Ting-En, Liu, Xiong, Wu, Yuchuan, Yang, Min, Wang, Minzheng, Zeng, Pengpeng, Gao, Lianli, and others. *arXiv preprint arXiv:2409.05840* (2024) [[link]](https://arxiv.org/pdf/2409.05840?)

- **SynthVLM: High-Efficiency and High-Quality Synthetic Data for Vision Language Models.** Liu, Zheng, Liang, Hao, Xiong, Wentao, Yu, Qinhan, He, Conghui, Cui, Bin, and Zhang, Wentao. *arXiv preprint arXiv:2407.20756* (2024) [[link]](https://arxiv.org/pdf/2407.20756)

- **World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering.** Wang, Jiacong, Wu, Bohong, Jiang, Haiyong, Xun, Zhou, Xiao, Xin, Guo, Haoyuan, and Xiao, Jun. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://arxiv.org/pdf/2409.20424?)

- **From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis.** Cheng, Chuanqi, Guan, Jian, Wu, Wei, and Yan, Rui. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (2024) [[link]](https://arxiv.org/pdf/2406.19934?)

- **Llm based generation of item-description for recommendation system.** Acharya, Arkadeep, Singh, Brijraj, and Onoe, Naoyuki. *Proceedings of the 17th ACM Conference on Recommender Systems* (2023) [[link]](https://www.academia.edu/download/107984813/3604915.pdf)

- **PMG: Personalized Multimodal Generation with Large Language Models.** Shen, Xiaoteng, Zhang, Rui, Zhao, Xiaoyan, Zhu, Jieming, and Xiao, Xi. *Proceedings of the ACM on Web Conference 2024* (2024) [[link]](https://dl.acm.org/doi/pdf/10.1145/3589334.3645633)

- **Llmrec: Large language models with graph augmentation for recommendation.** Wei, Wei, Ren, Xubin, Tang, Jiabin, Wang, Qinyong, Su, Lixin, Cheng, Suqi, Wang, Junfeng, Yin, Dawei, and Huang, Chao. *Proceedings of the 17th ACM International Conference on Web Search and Data Mining* (2024) [[link]](https://arxiv.org/pdf/2311.00423)

- **Large Language Models as Evaluators for Recommendation Explanations.** Zhang, Xiaoyu, Li, Yishan, Wang, Jiayin, Sun, Bowen, Ma, Weizhi, Sun, Peijie, and Zhang, Min. *arXiv preprint arXiv:2406.03248* (2024) [[link]](https://dl.acm.org/doi/pdf/10.1145/3640457.3688075)

- **Exploiting Asymmetry for Synthetic Training Data Generation: SynthIE and the Case of Information Extraction.** Josifoski, Martin, Sakota, Marija, Peyrard, Maxime, and West, Robert. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2303.04132)

- **Inpars-v2: Large language models as efficient dataset generators for information retrieval.** Jeronymo, Vitor, Bonifacio, Luiz, Abonizio, Hugo, Fadaee, Marzieh, Lotufo, Roberto, Zavrel, Jakub, and Nogueira, Rodrigo. *arXiv preprint arXiv:2301.01820* (2023) [[link]](https://arxiv.org/pdf/2301.01820)

- **READ: Improving Relation Extraction from an ADversarial Perspective.** Li, Dawei, Hogan, William, and Shang, Jingbo. *arXiv preprint arXiv:2404.02931* (2024) [[link]](https://arxiv.org/pdf/2404.02931)

- **STAR: Boosting Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models.** Ma, Mingyu Derek, Wang, Xiaoxuan, Kung, Po-Nien, Brantingham, P Jeffrey, Peng, Nanyun, and Wang, Wei. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29839/31460)

- **Adjudicating LLMs as PropBank Annotators.** Bonn, Julia, Madabushi, Harish Tayyar, Hwang, Jena D, and Bonial, Claire. *LREC-COLING 2024* (2024) [[link]](https://aclanthology.org/2024.dmr-1.pdf#page=126)

- **Annotated dataset creation through large language models for non-english medical NLP.** Frei, Johann, and Kramer, Frank. *Journal of Biomedical Informatics* (2023) [[link]](https://www.sciencedirect.com/science/article/pii/S1532046423001995)

- **ChatGPT as Your n-th Annotator: Experiments in Leveraging Large Language Models for Social Science Text Annotation in Slovak Language.** Hamerlik, Endre, {\v{S}}uppa, Marek, Bl{\v{s}}t{\'a}k, Miroslav, Kub{\'\i}k, Jozef, Tak{\'a}{\v{c}}, Martin, {\v{S}}imko, Mari{\'a}n, and Findor, Andrej. *Proceedings of the 4th Workshop on Computational Linguistics for the Political and Social Sciences: Long and short papers* (2024) [[link]](https://aclanthology.org/2024.cpss-1.6.pdf)

- **Zero-shot Cross-Lingual Transfer for Synthetic Data Generation in Grammatical Error Detection.** Latouche, Gaetan Lopez, Carbonneau, Marc-Andr{\'e}, and Swanson, Ben. *arXiv preprint arXiv:2407.11854* (2024) [[link]](https://arxiv.org/pdf/2407.11854)

- **A Causal Explainable Guardrails for Large Language Models.** Chu, Zhixuan, Wang, Yan, Li, Longfei, Wang, Zhibo, Qin, Zhan, and Ren, Kui. *arXiv preprint arXiv:2405.04160* (2024) [[link]](https://arxiv.org/pdf/2405.04160)

- **Zero-shot LLM-guided Counterfactual Generation for Text.** Bhattacharjee, Amrita, Moraffah, Raha, Garland, Joshua, and Liu, Huan. *arXiv preprint arXiv:2405.04793* (2024) [[link]](https://arxiv.org/pdf/2405.04793)

- **Text classification of column headers with a controlled vocabulary: leveraging LLMs for metadata enrichment.** Martorana, Margherita, Kuhn, Tobias, Stork, Lise, and van Ossenbruggen, Jacco. *arXiv preprint arXiv:2403.00884* (2024) [[link]](https://arxiv.org/pdf/2403.00884)

- **Self-Guide: Better Task-Specific Instruction Following via Self-Synthetic Finetuning.** Zhao, Chenyang, Jia, Xueying, Viswanathan, Vijay, Neubig, Graham, and Wu, Tongshuang. *First Conference on Language Modeling* (No Year) [[link]](https://arxiv.org/pdf/2407.12874)


## Assessing LLM-Generated Annotations


### Evaluating LLM-Generated Annotations

- **The turking test: Can language models understand instructions?.** Efrat, Avia, and Levy, Omer. *arXiv preprint arXiv:2010.11982* (2020) [[link]](https://arxiv.org/pdf/2010.11982)
  
- **Unnatural instructions: Tuning language models with (almost) no human labor.** Honovich, Or, Scialom, Thomas, Levy, Omer, and Schick, Timo. *arXiv preprint arXiv:2212.09689* (2022) [[link]](https://arxiv.org/pdf/2212.09689)

- **Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks.** Alizadeh, Meysam, Kubli, Ma{\"e}l, Samei, Zeynab, Dehghani, Shirin, Bermeo, Juan Diego, Korobeynikova, Maria, and Gilardi, Fabrizio. *arXiv preprint arXiv:2307.02179* (2023) [[link]](https://storage.prod.researchhub.com/uploads/papers/2024/04/24/2307.02179.pdf)

- **DISCO: Distilling Counterfactuals with Large Language Models.** Chen, Zeming, Gao, Qiyue, Bosselut, Antoine, Sabharwal, Ashish, and Richardson, Kyle. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (2023) [[link]](https://arxiv.org/pdf/2212.10534)

- **Codegen: An open large language model for code with multi-turn program synthesis.** Nijkamp, Erik, Pang, Bo, Hayashi, Hiroaki, Tu, Lifu, Wang, Huan, Zhou, Yingbo, Savarese, Silvio, and Xiong, Caiming. *arXiv preprint arXiv:2203.13474* (2022) [[link]](https://arxiv.org/pdf/2203.13474.pdf?trk=public_post_comment-text)

- **LMTurk: Few-shot learners as crowdsourcing workers in a language-model-as-a-service framework.** Zhao, Mengjie, Mi, Fei, Wang, Yasheng, Li, Minglei, Jiang, Xin, Liu, Qun, and Sch{\"u}tze, Hinrich. *arXiv preprint arXiv:2112.07522* (2021) [[link]](https://arxiv.org/pdf/2112.07522)

- **Large language models are zero-shot clinical information extractors.** Agrawal, Monica, Hegselmann, Stefan, Lang, Hunter, Kim, Yoon, and Sontag, David. *arXiv preprint arXiv:2205.12689* (2022) [[link]](https://arxiv.org/pdf/2205.12689.pdf?trk=public_post_comment-text)

- **Annollm: Making large language models to be better crowdsourced annotators.** He, Xingwei, Lin, Zhenghao, Gong, Yeyun, Jin, Alex, Zhang, Hang, Lin, Chen, Jiao, Jian, Yiu, Siu Ming, Duan, Nan, Chen, Weizhu, and others. *arXiv preprint arXiv:2303.16854* (2023) [[link]](https://arxiv.org/pdf/2303.16854)

- **Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge.** Wu, Tianhao, Yuan, Weizhe, Golovneva, Olga, Xu, Jing, Tian, Yuandong, Jiao, Jiantao, Weston, Jason, and Sukhbaatar, Sainbayar. *arXiv preprint arXiv:2407.19594* (2024) [[link]](https://www.rivista.ai/wp-content/uploads/2024/10/2407.19594v2.pdf)

- **Judging llm-as-a-judge with mt-bench and chatbot arena.** Zheng, Lianmin, Chiang, Wei-Lin, Sheng, Ying, Zhuang, Siyuan, Wu, Zhanghao, Zhuang, Yonghao, Lin, Zi, Li, Zhuohan, Li, Dacheng, Xing, Eric, and others. *Advances in Neural Information Processing Systems* (2023) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf)

- **From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge.** Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng and Huan Liu *arXiv preprint arXiv: 2411.16594* (2024) [[link]](https://arxiv.org/pdf/2411.16594)

- **CoEvol: Constructing Better Responses for Instruction Finetuning through Multi-Agent Cooperation.** Li, Renhao, Tan, Minghuan, Wong, Derek F, and Yang, Min. *arXiv preprint arXiv:2406.07054* (2024) [[link]](https://arxiv.org/pdf/2406.07054)

- **I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm.** Liang, Yiming, Zhang, Ge, Qu, Xingwei, Zheng, Tianyu, Guo, Jiawei, Du, Xinrun, Yang, Zhenzhu, Liu, Jiaheng, Lin, Chenghua, Ma, Lei, and others. *arXiv preprint arXiv:2408.08072* (2024) [[link]](https://arxiv.org/pdf/2408.08072?)


### Filtering & Selection

- **AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving.** Liang, Mingfu, Su, Jong-Chyi, Schulter, Samuel, Garg, Sparsh, Zhao, Shiyu, Wu, Ying, Chandraker, Manmohan. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2024) [[link]](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.html)

- **Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data.** Li, Yanda, Zhang, Chi, Yu, Gang, Wang, Zhibin, Fu, Bin, Lin, Guosheng, Shen, Chunhua, Chen, Ling, and Wei, Yunchao. *arXiv preprint arXiv:2308.10253* (2023) [[link]](https://arxiv.org/pdf/2308.10253)

- **SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization.** Kim, Hyunwoo, Hessel, Jack, Jiang, Liwei, West, Peter, Lu, Ximing, Yu, Youngjae, Zhou, Pei, Bras, Ronan, Alikhani, Malihe, Kim, Gunhee, and others. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2212.10465)

- **Aligning Large Language Models through Synthetic Feedback.** Kim, Sungdong, Bae, Sanghwan, Shin, Jamin, Kang, Soyoung, Kwak, Donghyun, Yoo, Kang, and Seo, Minjoon. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.13735)

- **AugESC: Dialogue Augmentation with Large Language Models for Emotional Support Conversation.** Zheng, Chujie, Sabour, Sahand, Wen, Jiaxin, Zhang, Zheng, and Huang, Minlie. *Findings of the Association for Computational Linguistics: ACL 2023* (2023) [[link]](https://arxiv.org/pdf/2202.13047)

- **Self-qa: Unsupervised knowledge guided language model alignment.** Zhang, Xuanyu, and Yang, Qing. *arXiv preprint arXiv:2305.11952* (2023) [[link]](https://arxiv.org/pdf/2305.11952)

- **Human-instruction-free llm self-alignment with limited samples.** Guo, Hongyi, Yao, Yuanshun, Shen, Wei, Wei, Jiaheng, Zhang, Xiaoying, Wang, Zhaoran, and Liu, Yang. *arXiv preprint arXiv:2401.06785* (2024) [[link]](https://arxiv.org/pdf/2401.06785)

- **Automated Construction of Theme-specific Knowledge Graphs.** Ding, Linyi, Zhou, Sizhe, Xiao, Jinfeng, and Han, Jiawei. *arXiv preprint arXiv:2404.19146* (2024) [[link]](https://arxiv.org/pdf/2404.19146)

- **Large Language Models Are Reasoning Teachers.** Ho, Namgyu, Schmid, Laura, and Yun, Se-Young. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (2023) [[link]](https://arxiv.org/pdf/2212.10071/pdf)

- **Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks.** Kang, Minki, Lee, Seanie, Baek, Jinheon, Kawaguchi, Kenji, and Hwang, Sung Ju. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/97faedc90260eae5c400f92d5831c3d7-Paper-Conference.pdf)

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** Wang, Xuezhi, Wei, Jason, Schuurmans, Dale, Le, Quoc V, Chi, Ed H, Narang, Sharan, Chowdhery, Aakanksha, and Zhou, Denny. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2203.11171.pdf?trk=public_post_comment-text)

- **Making Large Language Models Better Data Creators.** Lee, Dong-Ho, Pujara, Jay, Sewak, Mohit, White, Ryen, and Jauhar, Sujay. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2310.20111)

- **Reinforced self-training (rest) for language modeling.** Gulcehre, Caglar, Paine, Tom Le, Srinivasan, Srivatsan, Konyushkova, Ksenia, Weerts, Lotte, Sharma, Abhishek, Siddhant, Aditya, Ahern, Alex, Wang, Miaosen, Gu, Chenjie, and others. *arXiv preprint arXiv:2308.08998* (2023) [[link]](https://arxiv.org/pdf/2308.08998)

- **Raft: Reward ranked finetuning for generative foundation model alignment.** Dong, Hanze, Xiong, Wei, Goyal, Deepanshu, Pan, Rui, Diao, Shizhe, Zhang, Jipeng, Shum, Kashun, and Zhang, Tong. *arXiv preprint arXiv:2304.06767* (2023) [[link]](https://arxiv.org/pdf/2304.06767)

- **Selective In-Context Data Augmentation for Intent Detection using Pointwise V-Information.** Lin, Yen-Ting, Papangelis, Alexandros, Kim, Seokhwan, Lee, Sungjin, Hazarika, Devamanyu, Namazifar, Mahdi, Jin, Di, Liu, Yang, and Hakkani-Tur, Dilek. *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (2023) [[link]](https://arxiv.org/pdf/2302.05096)

- **GenSim: Generating Robotic Simulation Tasks via Large Language Models.** Wang, Lirui, Ling, Yiyang, Yuan, Zhecheng, Shridhar, Mohit, Bao, Chen, Qin, Yuzhe, Wang, Bailin, Xu, Huazhe, and Wang, Xiaolong. *The Twelfth International Conference on Learning Representations* (2023) [[link]](https://arxiv.org/pdf/2310.01361)

- **DISCO: Distilling Counterfactuals with Large Language Models.** Chen, Zeming, Gao, Qiyue, Bosselut, Antoine, Sabharwal, Ashish, and Richardson, Kyle. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (2023) [[link]](https://arxiv.org/pdf/2212.10534)

- **SASS: Self-Alignment with Semi-Supervised Instruction Data Generation.** Wang, Yue, Zhang, Haoke, Li, Juntao, Chang, Jinxiong, Zhang, Qishen, Liu, Zhongyi, Zhang, Guannan, and Zhang, Min. *No venue* (2023) [[link]](https://openreview.net/pdf?id=Q9vYgjcvrX)

- **Large Language Models Can Self-Improve.** Huang, Jiaxin, Gu, Shixiang, Hou, Le, Wu, Yuexin, Wang, Xuezhi, Yu, Hongkun, and Han, Jiawei. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2210.11610.pdf?src_trk=em6620554130ce70.4875613993116609)

- **West-of-N: Synthetic Preference Generation for Improved Reward Modeling.** Pace, Aliz{\'e}e, Mallinson, Jonathan, Malmi, Eric, Krause, Sebastian, and Severyn, Aliaksei. *arXiv preprint arXiv:2401.12086* (2024) [[link]](https://arxiv.org/pdf/2401.12086)

- **Self: Language-driven self-evolution for large language model.** Lu, Jianqiao, Zhong, Wanjun, Huang, Wenyong, Wang, Yufei, Mi, Fei, Wang, Baojun, Wang, Weichao, Shang, Lifeng, and Liu, Qun. *arXiv preprint arXiv:2310.00533* (2023) [[link]](https://arxiv.org/pdf/2310.00533)

- **Inpars-v2: Large language models as efficient dataset generators for information retrieval.** Jeronymo, Vitor, Bonifacio, Luiz, Abonizio, Hugo, Fadaee, Marzieh, Lotufo, Roberto, Zavrel, Jakub, and Nogueira, Rodrigo. *arXiv preprint arXiv:2301.01820* (2023) [[link]](https://arxiv.org/pdf/2301.01820)

- **DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature.** Li, Dawei, Yang, Shu, Tan, Zhen, Baik, Jae Young, Yun, Sunkwon, Lee, Joseph, Chacko, Aaron, Hou, Bojian, Duong-Tran, Duy, Ding, Ying, and others. *arXiv preprint arXiv:2405.04819* (2024) [[link]](https://arxiv.org/pdf/2409.13731)

- **Optimizing Language Model's Reasoning Abilities with Weak Supervision.** Tong, Yongqi, Wang, Sizhe, Li, Dawei, Wang, Yifan, Han, Simeng, Lin, Zi, Huang, Chengsong, Huang, Jiaxin, and Shang, Jingbo. *arXiv preprint arXiv:2405.04086* (2024) [[link]](https://arxiv.org/pdf/2405.04086)

- **Importance Weighting Can Help Large Language Models Self-Improve.** Jiang, Chunyang, Chan, Chi-min, Xue, Wei, Liu, Qifeng, and Guo, Yike. *arXiv preprint arXiv:2408.09849* (2024) [[link]](https://arxiv.org/pdf/2408.09849)


## LLM-Generated Annotations Utilization


### Supervised Fine-Tuning

- **AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving.** Liang, Mingfu, Su, Jong-Chyi, Schulter, Samuel, Garg, Sparsh, Zhao, Shiyu, Wu, Ying, Chandraker, Manmohan. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2024) [[link]](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.html)

- **Large Language Models Can Self-Improve.** Huang, Jiaxin, Gu, Shixiang, Hou, Le, Wu, Yuexin, Wang, Xuezhi, Yu, Hongkun, and Han, Jiawei. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2210.11610.pdf?src_trk=em6620554130ce70.4875613993116609)

- **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** Wang, Yizhong, Kordi, Yeganeh, Mishra, Swaroop, Liu, Alisa, Smith, Noah A, Khashabi, Daniel, and Hajishirzi, Hannaneh. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (2023) [[link]](https://arxiv.org/pdf/2212.10560)

- **Self: Language-driven self-evolution for large language model.** Lu, Jianqiao, Zhong, Wanjun, Huang, Wenyong, Wang, Yufei, Mi, Fei, Wang, Baojun, Wang, Weichao, Shang, Lifeng, and Liu, Qun. *arXiv preprint arXiv:2310.00533* (2023) [[link]](https://arxiv.org/pdf/2310.00533)

- **Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning.** Yang, Zhaorui, Liu, Qian, Pang, Tianyu, Wang, Han, Feng, Haozhe, Zhu, Minfeng, and Chen, Wei. *arXiv preprint arXiv:2402.13669* (2024) [[link]](https://arxiv.org/pdf/2402.13669)

- **Self-play fine-tuning converts weak language models to strong language models.** Chen, Zixiang, Deng, Yihe, Yuan, Huizhuo, Ji, Kaixuan, and Gu, Quanquan. *arXiv preprint arXiv:2401.01335* (2024) [[link]](https://arxiv.org/pdf/2401.01335.pdf?trk=public_post_comment-text)

- **Self-playing Adversarial Language Game Enhances LLM Reasoning.** Cheng, Pengyu, Hu, Tianhao, Xu, Han, Zhang, Zhisong, Dai, Yong, Han, Lei, and Du, Nan. *arXiv preprint arXiv:2404.10642* (2024) [[link]](https://arxiv.org/pdf/2404.10642)

- **Self-DC: When to retrieve and When to generate? Self Divide-and-Conquer for Compositional Unknown Questions.** Wang, Hongru, Xue, Boyang, Zhou, Baohang, Zhang, Tianhua, Wang, Cunxiang, Chen, Guanhua, Wang, Huimin, and Wong, Kam-fai. *arXiv preprint arXiv:2402.13514* (2024) [[link]](https://arxiv.org/pdf/2402.13514)

- **Stanford alpaca: An instruction-following llama model.** Taori, Rohan, Gulrajani, Ishaan, Zhang, Tianyi, Dubois, Yann, Li, Xuechen, Guestrin, Carlos, Liang, Percy, and Hashimoto, Tatsunori B. *No venue* (2023) [[link]](No link found)

- **Vicuna: An Open-Source Chatbot Impressing {GPT-4} with 90\%* ChatGPT Quality.** Chiang, Wei-Lin, Li, Zhuohan, Lin, Zi, Sheng, Ying, Wu, Zhanghao, Zhang, Hao, Zheng, Lianmin, Zhuang, Siyuan, Zhuang, Yonghao, Gonzalez, Joseph E., Stoica, Ion, and Xing, Eric P.. *No venue* (2023) [[link]](No link found)

- **Wizardlm: Empowering large language models to follow complex instructions.** Xu, Can, Sun, Qingfeng, Zheng, Kai, Geng, Xiubo, Zhao, Pu, Feng, Jiazhan, Tao, Chongyang, and Jiang, Daxin. *arXiv preprint arXiv:2304.12244* (2023) [[link]](https://arxiv.org/pdf/2304.12244.pdf?trk=public_post_comment-text)

- **Generating training data with language models: Towards zero-shot language understanding.** Meng, Yu, Huang, Jiaxin, Zhang, Yu, and Han, Jiawei. *Advances in Neural Information Processing Systems* (2022) [[link]](https://proceedings.neurips.cc/paper_files/paper/2022/file/0346c148ba1c21c6b4780a961ea141dc-Paper-Conference.pdf)

- **Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance.** Wang, Song, Tan, Zhen, Guo, Ruocheng, and Li, Jundong. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://arxiv.org/pdf/2311.01108)

- **PINTO: Faithful Language Reasoning Using Prompt-Generated Rationales.** Wang, PeiFeng, Chan, Aaron, Ilievski, Filip, Chen, Muhao, and Ren, Xiang. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2211.01562)

- **Distilling Reasoning Capabilities into Smaller Language Models.** Shridhar, Kumar, Stolfo, Alessandro, and Sachan, Mrinmaya. *Findings of the Association for Computational Linguistics: ACL 2023* (2023) [[link]](https://arxiv.org/pdf/2212.00193)

- **LogiCoT: Logical Chain-of-Thought Instruction Tuning.** Liu, Hanmeng, Teng, Zhiyang, Cui, Leyang, Zhang, Chaoli, Zhou, Qiji, and Zhang, Yue. *The 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.12147)

- **Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks.** Kang, Minki, Lee, Seanie, Baek, Jinheon, Kawaguchi, Kenji, and Hwang, Sung Ju. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/97faedc90260eae5c400f92d5831c3d7-Paper-Conference.pdf)

- **Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data.** Xu, Canwen, Guo, Daya, Duan, Nan, and McAuley, Julian. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2304.01196)

- **Exploiting Asymmetry for Synthetic Training Data Generation: SynthIE and the Case of Information Extraction.** Josifoski, Martin, Sakota, Marija, Peyrard, Maxime, and West, Robert. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2303.04132)

- **Inpars-v2: Large language models as efficient dataset generators for information retrieval.** Jeronymo, Vitor, Bonifacio, Luiz, Abonizio, Hugo, Fadaee, Marzieh, Lotufo, Roberto, Zavrel, Jakub, and Nogueira, Rodrigo. *arXiv preprint arXiv:2301.01820* (2023) [[link]](https://arxiv.org/pdf/2301.01820)

- **Code alpaca: An instruction-following llama model for code generation.** Chaudhary, Sahil. *Code alpaca: An instruction-following llama model for code generation* (2023) [[link]](No link found)

- **Code llama: Open foundation models for code.** Roziere, Baptiste, Gehring, Jonas, Gloeckle, Fabian, Sootla, Sten, Gat, Itai, Tan, Xiaoqing Ellen, Adi, Yossi, Liu, Jingyu, Remez, Tal, Rapin, J{\'e}r{\'e}my, and others. *arXiv preprint arXiv:2308.12950* (2023) [[link]](https://arxiv.org/pdf/2308.12950)

- **HuatuoGPT, Towards Taming Language Model to Be a Doctor.** Zhang, Hongbo, Chen, Junying, Jiang, Feng, Yu, Fei, Chen, Zhihong, Chen, Guiming, Li, Jianquan, Wu, Xiangbo, Zhiyi, Zhang, Xiao, Qingying, and others. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://arxiv.org/pdf/2305.15075)

- **Doctorglm: Fine-tuning your chinese doctor is not a herculean task.** Xiong, Honglin, Wang, Sheng, Zhu, Yitao, Zhao, Zihao, Liu, Yuxiao, Huang, Linlin, Wang, Qian, and Shen, Dinggang. *arXiv preprint arXiv:2304.01097* (2023) [[link]](https://arxiv.org/pdf/2304.01097)

- **Xuanyuan 2.0: A large chinese financial chat model with hundreds of billions parameters.** Zhang, Xuanyu, and Yang, Qing. *Proceedings of the 32nd ACM International Conference on Information and Knowledge Management* (2023) [[link]](https://arxiv.org/pdf/2305.12002)

- **Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.** Luo, Haipeng, Sun, Qingfeng, Xu, Can, Zhao, Pu, Lou, Jianguang, Tao, Chongyang, Geng, Xiubo, Lin, Qingwei, Chen, Shifeng, and Zhang, Dongmei. *arXiv preprint arXiv:2308.09583* (2023) [[link]](https://arxiv.org/pdf/2308.09583)

- **Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning.** Zhao, Haiteng, Liu, Shengchao, Chang, Ma, Xu, Hannan, Fu, Jie, Deng, Zhihong, Kong, Lingpeng, and Liu, Qi. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/129033c7c08be683059559e8d6bfd460-Paper-Conference.pdf)

- **Beyond Answers: Transferring Reasoning Capabilities to Smaller LLMsUsing Multi-Teacher Knowledge Distillation.** Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang and Nitesh V Chawla *arXiv preprint arXiv: 2402.04616* (2024) [[link]](https://arxiv.org/pdf/2402.04616)

### Alignment Tuning

- **Contrastive post-training large language models on data curriculum.** Xu, Canwen, Rosset, Corby, Del Corro, Luciano, Mahajan, Shweti, McAuley, Julian, Neville, Jennifer, Awadallah, Ahmed Hassan, and Rao, Nikhil. *arXiv preprint arXiv:2310.02263* (2023) [[link]](https://arxiv.org/pdf/2310.02263)

- **Aligning Large Language Models through Synthetic Feedback.** Kim, Sungdong, Bae, Sanghwan, Shin, Jamin, Kang, Soyoung, Kwak, Donghyun, Yoo, Kang, and Seo, Minjoon. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.13735)

- **West-of-N: Synthetic Preference Generation for Improved Reward Modeling.** Pace, Aliz{\'e}e, Mallinson, Jonathan, Malmi, Eric, Krause, Sebastian, and Severyn, Aliaksei. *arXiv preprint arXiv:2401.12086* (2024) [[link]](https://arxiv.org/pdf/2401.12086)

- **Learning Reward for Robot Skills Using Large Language Models via Self-Alignment.** Zeng, Yuwei, Mu, Yao, and Shao, Lin. *arXiv preprint arXiv:2405.07162* (2024) [[link]](https://arxiv.org/pdf/2405.07162)

- **SALMON: Self-Alignment with Instructable Reward Models.** Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David D. Cox, Yiming Yang, and Chuang Gan. *No venue* (2023) [[link]](https://openreview.net/pdf?id=xJbsmB8UMx)

- **Self-rewarding language models.** Yuan, Weizhe, Pang, Richard Yuanzhe, Cho, Kyunghyun, Sukhbaatar, Sainbayar, Xu, Jing, and Weston, Jason. *arXiv preprint arXiv:2401.10020* (2024) [[link]](http://readwise-assets.s3.amazonaws.com/media/wisereads/articles/self-rewarding-language-models/2401.10020.pdf)

- **Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation.** Zhang, Xiaoying, Peng, Baolin, Tian, Ye, Zhou, Jingyan, Jin, Lifeng, Song, Linfeng, Mi, Haitao, and Meng, Helen. *arXiv preprint arXiv:2402.09267* (2024) [[link]](https://arxiv.org/pdf/2402.09267)

- **Aligning Large Language Models by On-Policy Self-Judgment.** Lee, Sangkyu, Kim, Sungdong, Yousefpour, Ashkan, Seo, Minjoon, Yoo, Kang Min, and Yu, Youngjae. *arXiv preprint arXiv:2402.11253* (2024) [[link]](https://arxiv.org/pdf/2402.11253)

- **Optimizing Language Model's Reasoning Abilities with Weak Supervision.** Tong, Yongqi, Wang, Sizhe, Li, Dawei, Wang, Yifan, Han, Simeng, Lin, Zi, Huang, Chengsong, Huang, Jiaxin, and Shang, Jingbo. *arXiv preprint arXiv:2405.04086* (2024) [[link]](https://arxiv.org/pdf/2405.04086)

- **Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection.** Lee, Kyungjae, Hwang, Dasol, Park, Sunghyun, Jang, Youngsoo, and Lee, Moontae. *arXiv preprint arXiv:2403.14238* (2024) [[link]](https://arxiv.org/pdf/2403.14238)

- **Direct language model alignment from online ai feedback.** Guo, Shangmin, Zhang, Biao, Liu, Tianlin, Liu, Tianqi, Khalman, Misha, Llinares, Felipe, Rame, Alexandre, Mesnard, Thomas, Zhao, Yao, Piot, Bilal, and others. *arXiv preprint arXiv:2402.04792* (2024) [[link]](https://arxiv.org/pdf/2402.04792.pdf?utm_source=fbia)

- **Reinforced self-training (rest) for language modeling.** Gulcehre, Caglar, Paine, Tom Le, Srinivasan, Srivatsan, Konyushkova, Ksenia, Weerts, Lotte, Sharma, Abhishek, Siddhant, Aditya, Ahern, Alex, Wang, Miaosen, Gu, Chenjie, and others. *arXiv preprint arXiv:2308.08998* (2023) [[link]](https://arxiv.org/pdf/2308.08998)

- **Raft: Reward ranked finetuning for generative foundation model alignment.** Dong, Hanze, Xiong, Wei, Goyal, Deepanshu, Pan, Rui, Diao, Shizhe, Zhang, Jipeng, Shum, Kashun, and Zhang, Tong. *arXiv preprint arXiv:2304.06767* (2023) [[link]](https://arxiv.org/pdf/2304.06767)

- **Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping.** Wang, Haoyu, Ma, Guozheng, Meng, Ziqiao, Qin, Zeyu, Shen, Li, Zhang, Zhong, Wu, Bingzhe, Liu, Liu, Bian, Yatao, Xu, Tingyang, and others. *arXiv preprint arXiv:2402.07610* (2024) [[link]](https://arxiv.org/pdf/2402.07610)

- **Mixture of insighTful Experts (MoTE): The Synergy of Thought Chains and Expert Mixtures in Self-Alignment.** Liu, Zhili, Gou, Yunhao, Chen, Kai, Hong, Lanqing, Gao, Jiahui, Mi, Fei, Zhang, Yu, Li, Zhenguo, Jiang, Xin, Liu, Qun, and others. *arXiv preprint arXiv:2405.00557* (2024) [[link]](https://arxiv.org/pdf/2405.00557)

- **Iterative reasoning preference optimization.** Pang, Richard Yuanzhe, Yuan, Weizhe, Cho, Kyunghyun, He, He, Sukhbaatar, Sainbayar, and Weston, Jason. *arXiv preprint arXiv:2404.19733* (2024) [[link]](https://arxiv.org/pdf/2404.19733)


### Inference


- **Large Language Models are Human-Level Prompt Engineers.** Zhou, Yongchao, Muresanu, Andrei Ioan, Han, Ziwen, Paster, Keiran, Pitis, Silviu, Chan, Harris, and Ba, Jimmy. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://openreview.net/pdf?id=92gvk82DE-)

- **Auto-ICL: In-Context Learning without Human Supervision.** Yang, Jinghan, Ma, Shuming, and Wei, Furu. *arXiv preprint arXiv:2311.09263* (2023) [[link]](https://arxiv.org/pdf/2311.09263)

- **Empowering Large Language Models for Textual Data Augmentation.** Li, Yichuan, Ding, Kaize, Wang, Jianling, and Lee, Kyumin. *No venue* (No Year) [[link]](https://arxiv.org/pdf/2404.17642)

- **Self-generated in-context learning: Leveraging auto-regressive language models as a demonstration generator.** Kim, Hyuhng Joon, Cho, Hyunsoo, Kim, Junyeob, Kim, Taeuk, Yoo, Kang Min, and Lee, Sang-goo. *arXiv preprint arXiv:2206.08082* (2022) [[link]](https://arxiv.org/pdf/2206.08082)

- **Are Human-generated Demonstrations Necessary for In-context Learning?.** Li, Rui, Wang, Guoyin, and Li, Jiwei. *arXiv preprint arXiv:2309.14681* (2023) [[link]](https://arxiv.org/pdf/2309.14681)

- **Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations.** Chen, Wei-Lin, Wu, Cheng-Kuang, Chen, Yun-Nung, and Chen, Hsin-Hsi. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.15035)

- **Self-Demos: Eliciting Out-of-Demonstration Generalizability in Large Language Models.** He, Wei, Liu, Shichun, Zhao, Jun, Ding, Yiwen, Lu, Yi, Xi, Zhiheng, Gui, Tao, Zhang, Qi, and Huang, Xuanjing. *arXiv preprint arXiv:2404.00884* (2024) [[link]](https://arxiv.org/pdf/2404.00884)

- **Rephrase and respond: Let large language models ask better questions for themselves.** Deng, Yihe, Zhang, Weitong, Chen, Zixiang, and Gu, Quanquan. *arXiv preprint arXiv:2311.04205* (2023) [[link]](https://arxiv.org/pdf/2311.04205.pdf?trk=public_post_comment-text)

- **Dail: Data augmentation for in-context learning via self-paraphrase.** Li, Dawei, Li, Yaxuan, Mekala, Dheeraj, Li, Shuyao, Wang, Xueqi, Hogan, William, Shang, Jingbo, and others. *arXiv preprint arXiv:2311.03319* (2023) [[link]](https://arxiv.org/pdf/2311.03319)

- **Just rephrase it! Uncertainty estimation in closed-source language models via multiple rephrased queries.** Yang, Adam, Chen, Chen, and Pitas, Konstantinos. *arXiv preprint arXiv:2405.13907* (2024) [[link]](https://arxiv.org/pdf/2405.13907)

- **Self-Polish: Enhance Reasoning in Large Language Models via Problem Refinement.** Xi, Zhiheng, Jin, Senjie, Zhou, Yuhao, Zheng, Rui, Gao, Songyang, Liu, Jia, Gui, Tao, Zhang, Qi, and Huang, Xuan-Jing. *Findings of the Association for Computational Linguistics: EMNLP 2023* (2023) [[link]](https://arxiv.org/pdf/2305.14497)

- **Self-DC: When to retrieve and When to generate? Self Divide-and-Conquer for Compositional Unknown Questions.** Wang, Hongru, Xue, Boyang, Zhou, Baohang, Zhang, Tianhua, Wang, Cunxiang, Chen, Guanhua, Wang, Huimin, and Wong, Kam-fai. *arXiv preprint arXiv:2402.13514* (2024) [[link]](https://arxiv.org/pdf/2402.13514)

- **Large language models are zero-shot reasoners.** Kojima, Takeshi, Gu, Shixiang Shane, Reid, Machel, Matsuo, Yutaka, and Iwasawa, Yusuke. *Advances in neural information processing systems* (2022) [[link]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** Wang, Xuezhi, Wei, Jason, Schuurmans, Dale, Le, Quoc V, Chi, Ed H, Narang, Sharan, Chowdhery, Aakanksha, and Zhou, Denny. *The Eleventh International Conference on Learning Representations* (2022) [[link]](https://arxiv.org/pdf/2203.11171.pdf?trk=public_post_comment-text)

- **Universal self-consistency for large language model generation.** Chen, Xinyun, Aksitov, Renat, Alon, Uri, Ren, Jie, Xiao, Kefan, Yin, Pengcheng, Prakash, Sushant, Sutton, Charles, Wang, Xuezhi, and Zhou, Denny. *arXiv preprint arXiv:2311.17311* (2023) [[link]](https://arxiv.org/pdf/2311.17311)

- **Eliminating Reasoning via Inferring with Planning: A New Framework to Guide LLMs' Non-linear Thinking.** Tong, Yongqi, Wang, Yifan, Li, Dawei, Wang, Sizhe, Lin, Zi, Han, Simeng, and Shang, Jingbo. *arXiv preprint arXiv:2310.12342* (2023) [[link]](https://arxiv.org/pdf/2310.12342.pdf?fbclid=IwAR10rCAqJZCMGgMTuUZnUOqyNbpZ8rkF6b29Smdtlbjs8gTUQ3309vytK28)

- **It's Not Easy Being Wrong: Evaluating Process of Elimination Reasoning in Large Language Models.** Balepur, Nishant, Palta, Shramay, and Rudinger, Rachel. *arXiv preprint arXiv:2311.07532* (2023) [[link]](https://arxiv.org/pdf/2311.07532)

- **POE: Process of Elimination for Multiple Choice Reasoning.** Ma, Chenkai, and Du, Xinya. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2310.15575)


- **Self-refine: Iterative refinement with self-feedback.** Madaan, Aman, Tandon, Niket, Gupta, Prakhar, Hallinan, Skyler, Gao, Luyu, Wiegreffe, Sarah, Alon, Uri, Dziri, Nouha, Prabhumoye, Shrimai, Yang, Yiming, and others. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf)

- **Can LLMs Learn from Previous Mistakes? Investigating LLMs' Errors to Boost for Reasoning.** Tong, Yongqi, Li, Dawei, Wang, Sizhe, Wang, Yujia, Teng, Fei, and Shang, Jingbo. *arXiv preprint arXiv:2403.20046* (2024) [[link]](https://arxiv.org/pdf/2403.20046)

- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.** Chen, Wenhu, Ma, Xueguang, Wang, Xinyi, and Cohen, William W. *Transactions on Machine Learning Research* (2023) [[link]](https://arxiv.org/pdf/2211.12588)

- **Graph of thoughts: Solving elaborate problems with large language models.** Besta, Maciej, Blach, Nils, Kubicek, Ales, Gerstenberger, Robert, Podstawski, Michal, Gianinazzi, Lukas, Gajda, Joanna, Lehmann, Tomasz, Niewiadomski, Hubert, Nyczyk, Piotr, and others. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024) [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/29720/31236)

- **Reasoning with Language Model is Planning with World Model.** Hao, Shibo, Gu, Yi, Ma, Haodi, Hong, Joshua, Wang, Zhen, Wang, Daisy, and Hu, Zhiting. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (2023) [[link]](https://arxiv.org/pdf/2305.14992)

- **Tree of thoughts: Deliberate problem solving with large language models.** Yao, Shunyu, Yu, Dian, Zhao, Jeffrey, Shafran, Izhak, Griffiths, Tom, Cao, Yuan, and Narasimhan, Karthik. *Advances in Neural Information Processing Systems* (2024) [[link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf)
