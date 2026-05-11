Poe — A Style-Conditioned Language Agent 
A small language model agent fine-tuned to generate responses in the distinctive literary style of Edgar Allan Poe — exploring style
conditioning, prompt engineering, and lightweight language model deployment. 
 Overview 
This project investigates how language model behaviour can be steered toward a specific author’s style through a combination of: 
System-level style conditioning via structured prompting
Fine-tuning on curated author-specific text corpus
Response evaluation against style-specific criteria 
The goal is not to replicate Poe’s content, but to capture stylistic fingerprints: archaic vocabulary, gothic atmosphere, unreliable
narrator patterns, complex sentence structure, and rhetorical repetition. 
 Why This Problem? 
Style conditioning in language models is a proxy for a broader class of problems in controllable generation — making a model’s
outputs conform to specific constraints or distributions beyond the raw training data. 
The same techniques apply to: - Domain-specific technical writing assistants - Tone and register control in enterprise applications -
Persona-consistent dialogue agents - Low-resource style adaptation without full fine-tuning 
 Approach 
Base model (small LM)
↓
Style corpus (Poe's collected works — public domain)
↓
Fine-tuning / prompt conditioning
↓
Style-conditioned inference
↓
Evaluation: vocabulary overlap, sentence complexity, atmosphere scoring

 Poe’s Stylistic Fingerprints Targeted 
The model is conditioned to reproduce specific, measurable stylistic traits: 
Lexical: archaic pronouns (thee, thou, whilst), gothic vocabulary (sepulchre, phantasm, melancholy), elevated register 
Syntactic: long, subordinate-clause-heavy sentences; em-dash interruptions; parenthetical asides creating unreliable narrator effect 
Atmospheric: persistent themes of decay, obsession, the uncanny; first-person confessional voice; psychological unravelling 
 Project Status 
Code upload in progress. Architecture and training details to be added. 
 Quickstart 
# Requirements to be added with code upload
pip install transformers torch
 python agent.py --prompt "Tell me about the night" Limitations 
Small model size limits stylistic depth
Style consistency degrades on longer generations
No explicit style evaluation metric implemented (future work)
Prompt-based conditioning is fragile compared to full fine-tuning 
 Tools 
Python, HuggingFace Transformers, PyTorc
