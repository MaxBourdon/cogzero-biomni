RETRIEVER_PROMPT = """
Your task is to select ONLY the resources that are relevant to analyzing the provided single-cell data.

INPUT DATA:
{data_path}

AVAILABLE_RESOURCES
TOOLS:
{tools}

DATA LAKE ITEMS:
{data_lake}

SOFTWARE LIBRARIES:
{libraries}

CONTEXT
- You may use any available tools to inspect the data before selecting resources.
- Work step-by-step; include explanations or reasoning in each step.
- Keep in mind that **the state persists between code executions**: so if in one step you have created variables or imported modules, these will all persist.
- Return output in the format provided below.

DEFINITIONS
- Relevant (Direct): directly usable for this data expected analysis stages (e.g., SC RNA-seq QC, normalization, HVG, batch correction, dimensionality reduction, clustering, marker finding, differential expression, trajectory, pathway/GSVA/GSEA, cell type annotation, integration, visualization, reporting, export).
- Relevant (Indirect): supports or is required by a directly relevant resource (e.g., formats, IO, plotting, dependencies) or covers a likely optional stage (e.g., doublet detection, ambient RNA removal, cell cycle scoring, copy-number inference).
- Not Relevant: literature search or general knowledge tools unless they operate on THIS data; duplicate tools that provide no incremental capability; items that don’t support this data species/modality/file type.

DATA INSPECTION CHECKLIST
1) Load/peek metadata (n_cells, n_genes, assays/modalities, chemistry, species, batches, conditions, etc.).
2) File types and formats present (e.g., .h5ad/.loom/.mtx/.rds); required IO adapters.
3) Modality flags: scRNA-seq / snRNA-seq / scATAC / CITE-seq / spatial / multi-omics.
4) Known confounders: batches, ambient RNA, doublets; required corrections.
5) Downstream goals (if inferable from metadata): DE, trajectory, annotation, integration.

SELECTION RULES
A. Score each candidate (internally) and include if the total score ≥ 2:
   - Direct relevance (0–3)
   - Indirect relevance / dependency (0–2)
   - Pipeline coverage gain (0–2)
   - Duplication penalty (−1 if substantially overlapping with another chosen item)
   - Format/species mismatch penalty (−2)
B. Prefer a minimal non-redundant set that still covers the full likely analysis pipeline.
C. If nothing fits a category, return an empty list for that category.
D. Indices must be valid, unique, and sorted ascending — choose ONLY from the provided lists.

FINAL VALIDATION (perform internally before returning)
- Verify all indices exist in their respective lists.
- Remove duplicates and sort.
- Ensure every selected item contributes unique capability or required dependency.
- If two items are near-duplicates, keep the one with broader capability or better data fit.

OUTPUT SCHEMA
For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]
DATA_LAKE: [list of indices]
LIBRARIES: [list of indices]

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]

If a category has no relevant items, use an empty list, e.g., DATA_LAKE: []
"""
