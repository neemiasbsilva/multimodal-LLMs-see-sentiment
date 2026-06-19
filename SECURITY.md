# Security Policy

Thank you for helping keep this project secure. As a research-focused repository evaluating Multimodal Large Language Models (MLLMs), we take the security of our codebase and its users seriously.

---

## Supported Versions

Currently, only the latest commit on the `main` branch is actively supported with security updates. 

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| Older   | :x:                |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability in this project, **please do not report it by creating a public GitHub issue.** 

Instead, please report it privately by sending an email to:
**neemiasbuceli@alunos.utfpr.edu.br**

Please include the following information in your report:
* A description of the vulnerability and its impact.
* Detailed steps to reproduce the issue.
* (Optional) Any suggestions for a mitigation or fix.

We will acknowledge receipt of your vulnerability report within 48 hours and strive to provide a timeline for a fix or further updates shortly after.

---

## Scope of Vulnerabilities

Because this is an AI/Machine Learning research project, it is important to clarify what constitutes a security vulnerability for the purposes of private reporting.

### Out of Scope (Please use Public Issues)
The following are generally considered inherent limitations of current MLLMs or active research problems. Please report these by opening a standard **public GitHub issue** so the academic community can discuss them:
* **Adversarial Attacks:** Input images or text explicitly crafted to fool the model into producing incorrect sentiment.
* **Prompt Injection / Jailbreaking:** Techniques designed to bypass the model's standard prompt instructions.
* **Hallucinations:** The model generating confidently incorrect sentiment analyses.
* **Bias/Toxicity:** The model exhibiting biased behavior regarding specific demographic groups.

### In Scope (Please Report Privately)
The following are traditional software security flaws and should be **reported privately via email**:
* **Arbitrary Code Execution (RCE):** Vulnerabilities in the evaluation scripts or data loaders (e.g., unsafe deserialization of `pickle` files or untrusted model weights) that allow malicious code to run on the host machine.
* **Path Traversal:** Exploits allowing access to unauthorized files on the file system.
* **Dependency Vulnerabilities:** Critical security flaws in the specific pinned versions of third-party libraries (e.g., PyTorch, Transformers) used in our requirements files that have an active exploit path in our code.
* **Data Leaks:** Scripts that inadvertently expose sensitive local credentials or system variables.

---

Thank you for contributing to the safety and integrity of this research project!
