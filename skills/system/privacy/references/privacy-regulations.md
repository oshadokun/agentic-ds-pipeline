# Privacy Regulations Overview

This is a plain English overview for non-technical users.
It is not legal advice — always consult a qualified professional for your specific situation.

---

## GDPR (General Data Protection Regulation)
- **Where it applies:** European Union and EEA. Also applies to any organisation
  handling data of EU residents, regardless of where the organisation is based.
- **Key principles relevant to this pipeline:**
  - **Data minimisation:** Only collect and use the data you actually need
  - **Purpose limitation:** Only use data for the purpose it was collected for
  - **Storage limitation:** Don't keep data longer than necessary
  - **Security:** Protect data with appropriate technical measures
- **Personal data includes:** Names, email addresses, location data, IP addresses,
  cookie identifiers, health data, financial data, and anything that can identify
  a person directly or indirectly

## UK DPA 2018 (Data Protection Act)
- The UK's implementation of GDPR post-Brexit
- Applies to organisations handling personal data of UK residents
- Principles are substantially the same as GDPR

## CCPA (California Consumer Privacy Act)
- **Where it applies:** California, USA — affects businesses collecting data
  of California residents
- **Key rights:** Right to know what data is collected, right to delete, right
  to opt out of sale of personal data

---

## What This Means for Your Pipeline

### Before training a model on personal data, consider:
1. **Do you have a legal basis to use this data for modelling?**
   (e.g. consent, legitimate interest, contractual necessity)
2. **Is the data minimised?** — only the columns genuinely needed for the model
3. **Is the data secured?** — stored locally, not transmitted without awareness
4. **Are decisions made by the model consequential?** — automated decisions
   affecting people may have additional requirements under GDPR Article 22

### High-risk uses that require extra care:
- Models that make decisions about creditworthiness
- Models that make decisions about employment
- Models that make decisions about healthcare
- Models that use race, ethnicity, religion, health, or sexual orientation

### Safer practices built into this pipeline:
- Sensitive columns are identified and flagged before training
- User makes explicit decisions about sensitive data handling
- Credentials are never stored in plain text
- A privacy audit trail is maintained
- Data stays local unless user explicitly sends it elsewhere

---

## Anonymisation vs Pseudonymisation

### Anonymisation
- Data is truly anonymous if it cannot be re-identified by any means
- **Truly anonymous data is not subject to GDPR**
- Genuine anonymisation is hard — removing a name is not enough if
  other columns can identify the person (e.g. postcode + dob + gender)

### Pseudonymisation
- Replaces identifiers with codes — the same person always gets the same code
- The original data is not lost — just separated
- **Pseudonymous data is still personal data under GDPR**
- It is a security measure, not full anonymisation

### What this pipeline does
- "Pseudonymise" in this pipeline uses a one-way hash — the original
  value cannot be recovered from the code
- This provides stronger protection than standard pseudonymisation
- However, if the same person appears in a different dataset, their
  code will match — allowing re-identification across datasets
