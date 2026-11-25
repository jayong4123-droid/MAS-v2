# Statement 2: Employee feedback analysis

## 0. Data and cleaning (`outputs/s2_cleaning_audit.csv`)
155 responses, 4 departments. 18 non-responses (blank / "-" / "\n" only) and 4 one-word answers were set aside → **133 valid**.
Non-response by department: A 28%, B 23%, C 11%, D 0%. 17 valid answers look truncated (cut mid-sentence in the source).
Valid answers per department: **A 29, B 10, C 66, D 28**, so Dept B findings are anecdotal.

Style differs sharply by department: A reads like real, short civil-service comments; C mixes long generic LLM-style paragraphs,
numbered points and terse fragments; D is 28 near-uniform positive statements about team practice with zero non-response.
This looks like different sources/questions, so **department differences partly reflect how the data was produced**.
I report them but flag this wherever it matters.

## 1. What topics are in the responses? (`s2_company_topics.csv`, `s2_theme_stance.png`)
Share of valid responses mentioning each theme, with what is said about it (segment-level praise / suggestion / concern):

| Theme | % responses | Praise | Suggest | Concern |
|---|---|---|---|---|
| Teamwork & collaboration | 27 | 26 | 19 | 21 |
| Leadership, direction & transparency | 20 | 26 | 26 | 29 |
| Psychological safety & voice | 18 | **50** | 14 | 21 |
| Communication & information flow | 17 | 31 | 21 | 24 |
| Purpose, meaning & engagement | 16 | **48** | 4 | 26 |
| Resources, tools & processes | 15 | 25 | **38** | 25 |
| Workload & work-life balance | 14 | 17 | 20 | **37** |
| Pay, benefits & recognition | 14 | 20 | **35** | 20 |
| Line-manager support | 14 | 30 | **45** | 5 |
| Career growth | 13 | 23 | 27 | 23 |
| Diversity, inclusion & culture | 13 | 48 | 20 | 20 |
| Feedback & performance mgmt | 8 | 12 | **41** | 18 |
| Sustainability & social impact | 8 | 75 | 17 | 0 |
| Well-being & mental health | 5 | 11 | 33 | 33 |

Reading: staff praise team spirit, voice and purpose; **workload and leadership/transparency are where concern concentrates**;
pay, tools, feedback and manager support attract *constructive suggestions* ("more frequent feedback", "more investment in X") rather than complaints.

Method note: NMF on ~230 short segments (k scan 4-14 in `s2_k_scan.csv`; coherence is low and flat, typical for tiny short-text corpora) gave
usable but muddy topics (one catch-all absorbs "manager/goal/support"). I therefore use NMF as discovery/validation and a transparent
keyword taxonomy for final labels (`THEMES` in the code). Agreement NMI = 0.43; the taxonomy leaves 10% unassigned, mostly
truncated fragments. Keyword rules are a judgement call; I did not have labelled data to measure accuracy.

## 2. Topics by department (`s2_dept_theme_share/lift/qvalues_BH.csv`, `s2_dept_theme_heatmap.png`)
Permutation test (5,000 shuffles of department labels) per department × theme, Benjamini-Hochberg corrected across 56 cells.
**Only one cell survives: Dept D over-indexes on teamwork (54% vs 27% company, q=0.02)**, and given D's uniform, team-practice
style, that is probably the survey prompt, not culture. Descriptive, *not statistically established* patterns worth a follow-up:
- **A**: line-manager support (24%), psychological safety (28%), pay/recognition (21%) → managers and voice.
- **B** (n=10): leadership/transparency (50%), inclusion/culture (40%), voice (40%).
- **C** (largest): purpose/engagement (20%) and career growth (17%); highest concern share (0.30) → workload, career mobility, communication.
- **D**: teamwork and communication; lowest concern share (0.11).
Honest conclusion: with 133 responses, themes are largely **common across departments**; the data cannot support strong department-specific claims.

## 3. Individual profiles (bonus) (`s2_employee_profiles.csv`, `s2_personas.png`)
K-means on theme mix + tone + length gives 6 "response personas" (silhouette 0.12 → **weak separation, treat as descriptive**):
1. *Culture & inclusion voices* (15): diversity, culture, voice.
2. *Constructive suggesters* (27): mostly suggestions on communication, resources.
3. *Concern-led, short* (43): workload, career, leadership; mostly concern.
4. *Feedback & manager-focused* (11): performance feedback, line manager.
5. *Positive & engaged* (31): purpose, teamwork; praise 69%, concern 2%.
6. *Burnout/well-being* (6): well-being, workload (flag for HR follow-up).

What I deliberately do **not** infer: age, gender, ethnicity, seniority or any protected attribute: the data has none and text style is a poor, ethically risky proxy.
Profiles describe *what a person wrote*, not who they are, and should not be used to evaluate individuals.

## 4. Items for human review (not modelled)
Three responses contain allegations (hate crime, favouritism, "management games") that a survey owner should route to HR rather than treat as a "topic": ids 7979, 5858, 4895.
Tone is lexicon-based (transparent but unvalidated, e.g. "while X is competitive" can mislead) and a hand-labelled sample is the obvious next step.

## 5. Next steps
Sentence-embedding topic model (BERTopic) and an LLM labelling pass validated on a 100-segment hand-labelled set; confirm with the survey owner whether D answered a different question; collect more responses for B.
