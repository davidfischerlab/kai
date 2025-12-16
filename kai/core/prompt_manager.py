"""Centralized prompt management for all LLM interactions."""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from .utils import format_task_list

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext

# Global prompt manager instance
_prompt_manager_instance = None


def get_prompt_manager() -> 'PromptManager':
    """Get the global prompt manager instance."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance


class PromptScenario(Enum):
    """Different scenarios for LLM interactions."""
    AUTONOMOUS_MARK_COMPLETION = "autonomous_mark_completion"
    AUTONOMOUS_UPDATE_TASKS = "autonomous_update_tasks"
    AUTONOMOUS_UPDATE_CRITIQUE = "autonomous_update_critique"
    BACKTRACK_RECOVERY = "backtrack_recovery"
    CELL_SELECTION_ADDITION = "cell_selection_addition"
    CELL_SELECTION_DELETION_FOR_BACKTRACKING = "cell_selection_deletion_for_backtracking"
    CELL_SELECTION_REPLACEMENT = "cell_selection_replacement"
    CODE_FIXING = "code_fixing"
    CODE_FIXING_WITH_GUIDANCE = "code_fixing_with_guidance"
    CODE_UPDATE_WITH_GUIDANCE = "code_update_with_guidance"
    CODE_GENERATION = "code_generation"
    CODE_GENERATION_WITH_GUIDANCE = "code_generation_with_guidance"
    CODE_REVIEW = "code_review"
    ERROR_RECOVERY = "error_recovery"
    EXECUTION_MONITOR = "execution_monitor"
    AUTOLOOP_INTENT_CLASSIFICATION = "autoloop_intent_classification"
    INTENT_CLASSIFICATION = "intent_classification"
    QUESTION_ANSWERING = "question_answering"
    SECTION_CODE_REVIEW = "section_code_review"
    TASK_LIST_GENERATION = "task_list_generation"
    TASK_LIST_CRITIQUE = "task_list_critique"
    REASONING_CRITIQUE = "reasoning_critique"
    REFERENCE_WORKFLOW_SELECTION = "reference_workflow_selection"
    REFERENCE_WORKFLOW_SELECTION_ONLY = "reference_workflow_selection_only"
    REFERENCE_WORKFLOW_CELL_SELECTION = "reference_workflow_cell_selection"
    REASONING_RESPONSE_WITH_GUIDANCE = "reasoning_response_with_guidance"
    

class PromptManager:
    """
    Centralized prompt management with scenario-specific templating.
    
    Manages the generation of prompts for different scenarios by combining
    system prompts, user queries, RAG context, execution history, conversation
    history, and other contextual information into well-structured prompts
    for the LLM.
    
    This class ensures consistent prompt formatting across all LLM interactions
    and provides a single point of control for prompt engineering.
    
    Attributes:
        SHARED_SYSTEM_PROMPT: Base system prompt used for all scenarios
        SCENARIO_SYSTEM_PROMPTS: Additional prompts for specific scenarios
        SCENARIO_USER_TEMPLATES: User prompt templates for each scenario
    """
    
    # Shared system prompt for all scenarios
    SHARED_SYSTEM_PROMPT = """
You are a bioinformatics expert assistant who works on Jupyter notebook environment.
You help the user build reproducible analyses that follow best practices.
"""

    PROMPT_BUILDING_BLOCKS = {
        "count_matrices": """Working with single-cell gene expression matrices:
- Gene expression matrices are stored in anndata.AnnData instances and many basic operations are provided in the anndata and scanpy packages.
- These matrices are typically in adata.X, .layers and .raw.X.
  Note that several python libraries that use adata objects have API's to flexibly access any of these count matrices,
  so you often do not need to copy a matrix into the .X field to use it.
- They are usually supplied in sparse formats to optimize memory footprints.
    - Do not densify these matrices or perform operations that densify matrices.
    - Some older tutorials scale gene expresion count matrices, e.g. to unit variance. 
      Do not adapt this practice of scaling as it densifies the matrices, omit these steps when you adapt such code. 
      Often, you can just as well work with raw counts that are normlized and then log1p transformed, instead of scaling them.
- There are many different processing formats of expression matrices, here are a few key ones to be aware of:
    - ("raw") count matrices - integer valued positive counts - the direct output of many NGS read aligners.
    - normalized - expression values in cells (across genes) are normalized to sum to a constant (often 10000), i.e. divided the sum of all counts in that cell and multiplied by that constant, e.g by sc.pp.normalize_total
    - log - expression value are log1p transformed, often with sc.pp.log1p, note that when people talk about log transform on these matrices, they almost always mean log1p
    - log-normalized - normalized and then log1p transformed, this is a very common processing format
    - any from the above and further modified: for example scaled, batch corrected, etc.
- Unless otherwise specified in a workflow, treat log-normalization as a standard preprocessing of raw count matrices that can be used for analyses and plotting.
  log-normalization is starting from a raw count matrix, normalizing each cell to sum up to a constant (usually 10000), and then log1p transforming it.
- Not all formats are interconvertible. For example, if you are only given a log-normalized matrix, you cannot recover the count matrix anymore if the normalization factors of each cell (size factors) are not saved in the object.
  You need to consider this when adapting workflows that depend on particular formats and find workflows that can use the formats that you have or document violation of formats transparently.
- The full set of protein coding genes is around 20000 for both mouse and human, in some datasets, up to around 60000 genes are reported that include non-protein-coding genes that are typically much lower expressed.
  The majority of these genes are typically lowly or not variably-expressed: typically 2000-6000 genes display significant variation above expected technical variation across cells in a sample but this depends heavily on the sample.
  For compute-intensive or noise-sensitive tasks, it often makes sense to focus on highly variable genes to improve run time or reduce the contribution of measurement noise the modeled variation.
  In particular, this is relevant for modeling or attributing variation per gene, for example with standard statistical models or neural networks with reconstruction losses, but also for correlation measures.
- Good management of processing formats of gene expression matrices is one of the most crucial parts of working with this data.
    - Know what you are working with: make sure you understand what processing formats you have access to.
    - Document how matrices are transformed across tasks, e.g. annotate in task description which fields you access or which transformations you introduce.
    - Keep manipulations of expression matrices minimal: As far as possible, you should try to generate the representations of the expression matrices that are relevant in the first few tasks and then access these matrices downstream.
      For example: 
        * avoid copying raw count matrices back and forth between .raw and .X
        * avoid generating copies of adata instances just to move a matrix around if not necessary because of API restrictions,
        * avoid applying and reverting transformations repeatedly - e.g. keep a count and a log1p normalized matrix in an anndata instead of creating one from the other in multiple tasks,
        * keep data formats in .raw that would allow you to recover the other matrices so that you can recover if you make a mistake and can keep these processing choices transparent
      This does not include subsetting - subsetting can often occur throughout a workflow.
    - Check processing formats when adapting workflows: Often, reference workflows and tools indicate which formats they expect (e.g. counts or normalized data).
      Highlight these choices in task definitions so that they do not get lost downstream.

Provenance in interpreting gene expression:
Gene expression data is often interpreted with respect to existing knowledge, examples include marker genes, signatures, cell types expected in a particular tissue, etc.
If you introduct such or similar knowledge to interpret data, it is important that you cite sources that support this knowledge.
For example, this can be parts of reference workflows.
Describe these knowledge sources in concise comments in the code where you use them.
""",

        "task_list_description": """The task list should be designed as follow:
- It should break the user query down into individual analysis tasks.
- Each task consists of the following fields:
    * "id" (integer): 1-based indexing in all tasks
    * "task" (string): description of task
    * "status" ("pending", "active", or "completed" - initially, all are "pending")
- Tasks are either standard (i.e. coding) tasks or reasoning tasks (which are identified by having their task description prefixed with "[reasoning]").
  Only label tasks with the prefix "[reasoning]" if they fit the criteria for reasoning tasks (as outlined below in "Reasoning tasks"): 
    * tasks that generate markdown cells
    * tasks that do not contain new analyses that would require code to be executed
    * tasks that do not require code to be performed
  All other tasks do not receive a dedicated label and will produce and execute code.
- Each coding task is a concise description of an analysis that will serve as guidance for building code.
- Approximately, each coding task should map to one jupyter notebook cell as it will be used to generate the code for one cell.
  The task list represents the flow of the overall analysis and its discretization into individual tasks.
  The size of tasks in a task list should be determined by points in the workflow at which the analysis needs to be inspected:
  When outputs need to be interpreted or analyses need to be sanity checked, this is best done through the output field of a jupyter notebook cell and should correspond to the end of one task.
  Do not make tasks smaller then necessary because tasks lists with fewer items are easier to handle.
  Try to combine small analyses (e.g. single function calls) into one task if they belong to a single coherent analysis unit and if you can sanity check them together.
  This both often results in more concise and accessible code, and in more efficient sanity checks.
- Task descriptions do not contain ANY code, ideally they are centered on concepts and ideas rather than specific implementations so that the implementation can be adapted to overall analysis progess.
- When sequentially performed in a jupyter notebook, these tasks should address the objective.
- If you include analyses that depend on particular inputs, often fields in adata.obs or adata.var or certain count or normalization formats of gene expression fields that are not set during this anylsis,
  account for checking these fields in the relevant objects in the task list so that the subsequent tasks can access the output of these format discovery tasks.
- Plan analyses that complements the notebook, do not repeat imports, analyses or code that is already there.
- Further analyses may be added to the notebook later, so the planned analyses should strictly only address the objective and not explore other analysis options.
- Unless otherwise stated, this is an intermediate part of the full notebook and you should not save any anndata objects or other data structures from this analysis to disk.

Planning tasks that require interpretion of previous analyses:
- If later tasks in the plan depend on the outcomes of earlier tasks, account for that interpretation in the task rather than assuming a specific outcome.
  This task list will be refined after each step and code generation based on this task list will have access to outputs of previous analyses.
  Therefore, you only need to describe what output needs to be interpreted for a particular analysis.
- For example:
    * If you need to consider the outout of a previous task to implement a subsequent task, 
      describe that in the task description of that task that generates the output and the task and depends on it.
    * If you need to query an API to inform an implementation, account for that as a distinct task and describe how it informs the later task that depends on it.
- Make sure that the key outputs that are necessary for interpretation are not only provided as plots because the interpreting agents will not be able to process plots.
  Include plots because they are useful for human analysts reading the analysis, but provide key results also as outputs in text form that can directly be interpreted by agents.
  Include clear descriptions of outputs that are necessary for interpretation in the description of tasks that are relevant for interpretation to guide code generation for these tasks.
- Make sure that the interpretation logic is flawless and documented: how can one interpret a particular input and how does this result support a decision in a later task.
- For outputs that need to be interpreted later, it is often useful to provide intermediate summaries and interpretations of analyses in text form so that subsequent task can access these results more easily.

Reasoning tasks:
In many analysis workflows, there are key decision making or interpretation steps at which we need to reason using the intermediate analysis results.
Here, we consider reasoning as a text-based operation: in a reasoning tasks, an LLM will generate a text description of the results that are considered from previous tasks and an interpretion with respect to a particular question or problem.
Because it is a text-based operation (text in, text out), a reasoning task cannot include steps that require code excution - it is based on previous tasks that provide results from code executions.
This reasoning is then available to subsequent tasks because it is recorded in the notebook, effectively compartmentalizing a complex problem into a chain-of-thoughts along cells in the notebook.
Adhere to the following regarding reasoning tasks:
- Label tasks as such dedicated reasoning tasks that will output markdown cells by prefixing the task description with "[reasoning]".
- In the task description, make sure that the reasoning problem is very clearly defined so that the output is actionable.
- The following features may justify creating a reasoning task:
    * the presence of interpretation information in reference workflows, 
    * the user may have suggested reasoning steps by indicating decision/interpretation/reasoning etc. at particular points in the analysis,
    * or you may include them because you think they are necessary to complete the logical flow of the analysis or are useful to prevent subsequent hallucinations
- The following features may justify labeling a task as a reasoning task:
    * The task description only includes reasoning and interpretation operations and does not require any code to be executed.
- The following disqualify a task from a being a reasoning task:
    * The task contains statements that require code to be executed: reasoning tasks will be markdown cells and cannot execute any code.
    * The task is a summarization step that is not text centric, for example selecting based on metadata or generating summary tables from data containers -
      these are tasks that are often better handled by dedicated summarization code in a standard task. 
      If necessary, such summarizing standard tasks can be followed by reasoning tasks that interpret these data summaries.

Further guidance that is applicable if you have access to reference workflows (in "Session context: Reference workflows"):
You should use reference workflows as building blocks of an analysis that addresses the user's query.
Each useful reference workflow typically covers multiple tasks and may allow you to adapt:
    * the usage of dedicated python libraries that cover key parts of analyses requested by the user, 
    * specific applications, e.g. tissues, perturbations, diseases etc., 
    * or analysis objectives that may be addressed in more general tutorial notebooks.
These reference notebooks are published and of high quality so you should prefer refering to them rather than designing analyses from scratch.
Adhere to the following rules and guidelines:
- Use reference workflows for every task in the list unless that is not reasonable / you don'd have an appropriate reference.
- One jupyter notebook cell in the reference notebook does not need to correspond to one task in this task list:
  for example, if a cell in a reference notebook does not produce an output that needs to be interpreted or does not need to be sanity checked, 
  you can merge it with adjacent cells from references into single coherent tasks to keep the task list more compact.
  Use the general guidance on task list design from above to decide the size of tasks when adapting from references.
- Reference workflows often produce panels/figures as intermediate outputs as a means for the analyst to interpret the results
  In contrast to these reference workflows that are iteratively executed by humans, this task list will be handled by an agent that cannot interpret figures.
  Do still include them as they will help humans observe your work, but you need to account for numeric / text based outputs of tasks in those cases that are accessible as intermediate results for agents.
- Reference workflows sometimes load content from local files. Unless you have access to these files, you cannot perform these steps. If the files are downloaded through an API, you can choose to perform these steps.
- IMPORTANT: For each task that you can map to or derive from a reference notebook, you must insert a citation at the end of the task description:
  the reference 'Notebook ID' and specific indices of cells ('cell_indices') in the format: "[adapted from: 'Notebook ID', cells: cell_indices]".
  This citation must be exactly to the selection of cells from selected notebooks that you have access to in "Session context: Reference workflows",
  you cannot cite any cells that are note listed there.
  If a task does not relate to a reference notebook, insert "[custom step]" instead of this citation.
- If a workflows does not perfectly describe the analysis you are looking for but is still a useful reference,
  you can cite and explain the necessary adaptation in the task description.
  For example, you may not find a reference for a particular tool with the exact hyperparameters or settings requested by the user,
  but a reference for general usage of the tool may still be useful to then modify this reference to the user's specifications.
  This is preferable to creating custom steps if the reference your are using is meaningfully related to the task, e.g. describes general or slightly deviating usage of the requested tool.
""",

        "task_list_update_rules": """- You are working on a carefully thought through analysis plan so only deviate from it or extend it if absolutely necessary, in particular if you are deleting tasks.
  Your default is to not update the task list - only do so if issues or results from previous tasks raise strong concerns that the planned tasks are no longer appropriate and need to be changed for the analysis to succeed.
- You can only update tasks that are not "completed".
- You may modify or remove "pending" tasks or insert new "pending" tasks in the list "pending" tasks.
  Note that this modification must be inline with the overall analysis objective "orignal_user_prompt".
- Try to avoid extending the task list when it's almost fully completed, particularly for open-ended queries; 
  instead, prefer completing the current analysis plan so that it can be used to guide future analyses.
- In some cases, "recovery_objective" may be provided which means that the system backtracked to an earlier point in the anaylsis by setting previously "completed" tasks to "pending".
  In these cases, reconsider carefully how you can change the task list to overcome the issues that were encountered. 
  This would often involve more substantial changes to the task list as compared to a normal update.
- If you see in the execution history section, that a task repeatedly fails and there is no clear path to fixing it, consider replacing or modifying that task.
""",

        "code_response": """Adhere to these rules:
- Use python code.
- Use functionalities provided by specialized omics libraries where possible, e.g. scanpy plotting for plots based on anndata objects.
- Provide compact code with little formatting so that cells are easily readable at a glance, only providing minimal comments where necessary.
- Key outputs should not only be provided as plots, while useful for humans and should therefore be included, they cannot be interpreted by agents. Instead, also provide text-based summaries of these key results, e.g. statistics, tables, etc.
  Note that large tables are truncated in the jupyter cell output - opt for relevant summary statistics of such tables as outputs:
  For example, pandas dataframes with several rows are truncted to the first few and last few rows in the output. 
  To make sure that this output is fully accessible for downstream interpretation, modulate pd.options.display.max_rows or find other ways to present key aspects of the data as text output.
- When the analysis objective depends on outputs of previous analyses in the workflow, you need to interpret these outputs.
  You have a tendency to skip this interpretation and to hallucinate code at these positions. You cannot do that.
  To guide code writing, consider carefully which parts of the new code relate to previous parts of the notebook (either code or code outputs).
  If you cannot find the information that you require in the previous analysis outputs, you need to create these outputs so that they can be interpreted in the next step.
  
Best practices: 
Adhere to these best practices if reasonable:
- If reasonable, print key modification outcomes at the end of cells so that one can sanity check success based on the output of that cell: 
  for example, when subsetting an anndata instance, print the anndata instance to control shape, or when an attribute was set in an anndata instance, check that this slot was meaningfully filled.
  Note that these finalizing lines should not discrupt the analysis flow of the notebook, only add them if they are minimal and not disruptive.
- To summarize or output analyses, you would often see plots being used in reference code that you might have access to. 
  Use plots to communicate results to human analysist where possible but keep in mind that you will not be able to interpret plots in the next iteration: you can only read text.
  Therefore, also include text-based outputs, e.g. summary tables where numeric results are important, where reasonable.
- If you need to check the state expression matrices, e.g. adata.X, .raw.X or .layers, for normalization states, produce code that tells you if the data is in count format, normalized, log-normalized, scaled etc.
  Consider using the following snippets for the first few cells (accounting for sparse matrices by adding .todense() on a slice of the first few cells only):
        - "x[:5, :].sum(1)" to check if the sum of all values for any one cell (out of a subset) sums to the same constant which would indicate that the data is normalized
        - "np.expm1(x[:5, :]).sum(1)" to check if the sum of all expm1 transformed values for any one cell (out of a subset) sums to the same constant which would indicate that the data is log-normalized
        - "(np.min(x[:5, :]), np.max(x[:5, :]))" to check for negative values and to guess is raw counts or transformed
        - etc.
- The full set of protein coding genes is around 20000 for both mouse and human, in some datasets, up to around 60000 genes are reported that include non-protein-coding genes that are typically much lower expressed.
  The majority of these genes are typically lowly or not variably-expressed: typically 2000-6000 genes display significant variation above expected technical variation across cells in a sample but this depends heavily on the sample.
  For compute-intensive or noise-sensitive tasks, it often makes sense to focus on highly variable genes to improve run time or reduce the contribution of measurement noise the modeled variation.
  In particular, this is relevant for modeling or attributing variation per gene, for example with standard statistical models or neural networks with reconstruction losses, but also for correlation measures.
- Unless absolutely necessary, do not update or download data or models that are already cached to reduce run time.

Adapting reference workflows:
Adhere to these guidlines if you are given reference workflows:
- These reference notebooks are published and often of high quality. Therefore, you should base generated code closely on these references.
- If you are adapting code that outputs panels as a means for the analyst to interpret the results, you need to also produce numeric outputs in text form that represent analysis results that the analyst would look for in the panel.
  This is because in contrast to these reference workflows that are iteratively executed by humans, your code is handled by an agent that cannot interpret figures.

Response Formatting:
- Format code using markdown code blocks: ```python and ```
- Provide a single solution (not multiple suggestions).
""",

        "task_list_update": """In most cases, you should keep the current task list.
If you want to keep the current tasks, set the output field "update_rule" to "KEEP" and provide an empty string in "update_rationale", and return an empty list in the "tasks" field.
If you want to update any tasks, set the output field "update_rule" to "UPDATE" and provide reasoning in "update_rationale", and provide the new non-completed tasks in the output field "tasks".
The non-completed tasks will be appended to the existing completed tasks and should be marked with status "pending".
In this case, reason why you need to change the task list overall and explain changes to each single task that you are modifying in "update_rationale".
Do not change tasks for which you cannot explain why you are changing them.
""",

        "task_list_update_critiqued": """Improve the updated version of the task list based on this feedback.
The non-completed tasks will be appended to the existing completed tasks and should be marked with status "pending".
Reason why you need to change the task list overall and explain changes to each single task that you are modifying in "update_rationale".
Do not change tasks for which you cannot explain why you are changing them.
""",

        "task_list_update_output": """
Output the following fields:
- "tasks": This is the task list. The value of this field is a list of tasks where each task is a dictonary with the fields:
            * "id" (integer): 1-based indexing in all tasks
            * "task" (string): description of task
            * "status": set to "pending" for all tasks
- "retrieval_queries": If the current reference workflow selection is not sufficient, you can supply "retrieval_queries" - a list of string queries that will then be used to find further reference workflow candidates that you can then use in the next step.
  This is typically applicable if key tasks are not covered by reference workflows and are implemented as custom steps instead.
  If tasks are already adapted from reference workflows, but those references are not focussed on the analysis that you are adapting (for example because they use it as a small part of a larger analysis),
  query more focussed focussed reference workflows, e.g. tutorials dedicated to the analysis or tool you want to implement.
  For standard algorithms, for example commonly used unsupervised methods implemented in scanpy, search for general best practice or tutorial workflows that cover entire analyses rather than attempting to find specific tutorials.
  Aim for a 1-2 sentence summary of the content you are looking for per query, this query will be processed with a sentence embedding model and queried against a database.
  If and only if you think that the selection of reference workflows is sufficient to address the user query, you can skip this output.
- "update_rationale": Reasoning for performing the update.
- "update_rule": Return "KEEP" or "UPDATE".
""",

        "task_list_update_output_critiqued": """
Output the following fields:
- "tasks": This is the task list. The value of this field is a list of tasks where each task is a dictonary with the fields:
            * "id" (integer): 1-based indexing in all tasks
            * "task" (string): description of task
            * "status": set to "pending" for all tasks
- "retrieval_queries": If the current reference workflow selection is not sufficient, you can supply "retrieval_queries" - a list of string queries that will then be used to find further reference workflow candidates that you can then use in the next step.
  This is typically applicable if key tasks are not covered by reference workflows and are implemented as custom steps instead.
  If tasks are already adapted from reference workflows, but those references are not focussed on the analysis that you are adapting (for example because they use it as a small part of a larger analysis),
  query more focussed focussed reference workflows, e.g. tutorials dedicated to the analysis or tool you want to implement.
  For standard algorithms, for example commonly used unsupervised methods implemented in scanpy, search for general best practice or tutorial workflows that cover entire analyses rather than attempting to find specific tutorials.
  Aim for a 1-2 sentence summary of the content you are looking for per query, this query will be processed with a sentence embedding model and queried against a database.
  If and only if you think that the selection of reference workflows is sufficient to address the user query, you can skip this output.
- "update_rationale": Reasoning for performing the update.
- "update_rule": You are already in updating mode, return "UPDATE".
""",

}

    PROMPT_SECTION_SUMMARIES = {
        "context_sections_heading": """=== Further context:
In the following, you will find context sections that you can use to inform your code generation.
These sections are divided by headings prefixed with '=== Session context:'.
""",

        "context_sections_rag": """- If enabled, a retrieval section of related API documentation or usage code snippets that can help using specialized python libraries. Note that code snippets are not guaranteed to be up-to-date.""",

        "context_sections_reference_workflow_content": """- If enabled, a section with retrieved reference workflows. These workflows were selected based on their relevance for the overall analysis and may contain further documentation concerning their adaptation to this scenario.""",

        "context_sections_conversation_history": """- A conversation history section that you can use to contextualize the current user query.""",

        "context_sections_execution_history": """- An execution history section that contains the last executed cells - this can help you understand recent activity in the notebook.""",

        "context_sections_notebook_structure": """- An notebook structure section that contains all cells in the notebook - this can help you plan actions in the context of the full notebook."""
    }

    # Scenario-specific system prompt additions
    SCENARIO_PROMPTS = {
        "autonomous_mark_completion": f"""You are given a task list and will update completion status of tasks that were completed in the last notebook execution step.

Update the task completion status based on executed code and results.

Adhere to these critical rules:
1. Return the status for ALL tasks in the list, do not change the number of tasks.

2. The new status that you assign must be one of the following: "pending" or "completed".
Note that in the input task list that you receive, also "active" exists, but this is a label you will remove as outlined below.

3. No completed tasks can appear after pending tasks (maintain logical order).

4. Perform updates only based on the following decision scheme:
  
4a) The "active" task completed without error and was sufficiently addressed. 
    Consider code built and interpret execution results for code cells, and reasoning for markdown cells. 
        * Code cells: The code in the last cell(s) (see execution history) needs to address the task for it to be sufficiently addressed.
          In addition, the outputs of these cells (see execution history) need to to support that the analysis was succesful for the task to be sufficiently addressed
        * Markdown cells: The reasoning needs to be valid and address the task for it to be sufficiently addressed.
          Do not reject reasoning with minor improvements or suggestions, only if you see errors or hallucinations.
    Set the "active" tasks to "completed". Leave "retry_objective": null. Leave "recovery_objective": null.
    Do not set "pending" tasks to "completed" even if they were addressed.

4b) The "active" task is 
        * Code cells: The cell did not complete without error or was not sufficiently addressed as defined in 4a), but the issue can be overcome by replacing the cell with new code and executing this new cell.
          This is the case if either, you found an execution error in the last cell that can be fixed by modifying the cell that caused it, e.g. a syntax error,
          or, you interpreted the output of the last executed cells and concluded that the code in that cells did needs to be modified to sufficiently address the task.
          The latter includes cases in which the analysis in the cell was correct but the output did not meet the output requirements,
          for example, if a key output for downstream interpretation was not fully or succesfully shown in text form.
        * Markdown cells: A reasoning task but was not sufficiently addressed as defined in 4a).
    If 4b) is chosen, in the next step, the active task will be re-attempted.
    Leave the "active" tasks as "active". Provide a short "retry_objective" explaining what needs to change. Leave ^"recovery_objective": null.
    Note on "retry_objective": reasoning tasks only have markdown cell output, other tasks have code cell output.

4c) Neither 4a) nor 4b) is applicable because you do not think that the active task should be pursued any longer - 
    often this means you discovered a fundamental flaw in the task list and need to revise tasks that are "completed", or you encounter a repeated error that cannot be recovered via 4b).
    In this case, you may backtrack by setting the active and if necessary also earlier tasks (up to and including the "active" tasks), back to "pending".
    Do not use 4c if you want to revise a resoning task only, use 4b instead.
    In the next step, these tasks will be updated, so backtrack up to the point in the analysis that you consider to still have been valid.
    Leave "retry_objective": null. Provide a short "recovery_objective" explaining what needs to change.
    Use this option carefully and only when necessary.
""",

        "autonomous_update_tasks": f"""You are given the current task list that addresses a user query (see conversation history) and will consider updating definitions of non-completed tasks.
Consider all rules and recommendations on task list design that are given in the following, make sure you do not ignore any of them.

Any updates should conserve the following design criteria of the list:
====== This is the start of the design criteria for the task list:
{PROMPT_BUILDING_BLOCKS['task_list_description']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
====== This is the end of the design criteria for the task list.

In addition, the changes the you introduce must adhere to these rules:
{PROMPT_BUILDING_BLOCKS['task_list_update_rules']}
""",

        "backtrack_recovery": """A part of the anaylsis was removed. Analyze the recovery objective that outlines planned changes and the observed errors to determine if the pythons session needs to be restarted.

A restart is necessary if the deleted code and error did not cause any non-recovarable change to key objects in the session or declared large objects that are no longer needed.
For example, a restart is necessary if any of the following apply:
- a subsetting of or a modification of gene expression features or medata of an anndata instance may not be recoverable
- overwriting or setting any of the expression matrices in an anndata instance, including .X, .layers, and .raw
- a large object was declared in the cell, e.g. new adata instance, but will not be used any more and will just clog memory
A restart is not necessary if the above do not apply, for example in these cases:
- a failed modification of metadata that did not introduce changes
- failed usage of a tool
""",
        
        "cell_selection_addition": """You select a cell number (target_cell) after which to add a new cell in a Jupyter notebook based on a user query.

Use 0-based indexing to select the cell number from the notebook structure (see notebook structure section).
For empty notebooks (0 cells), use target_cell = -1 to indicate "insert at the beginning".
In all other cases, distinguish the following intents to guide the choice of target_cell.

**MOVE_TO_NEXT_FROM_LAST_EXECUTED**: User wants to add a cell after the last executed cell.
- The user wants to add cells after the last executed cell.
- This is the default choice, only deviate from this choice if you see good evidence for one of the following intents.

**MOVE_TO_NEXT_FROM_CURSOR**: User wants to add a cell after the currently selected cell.
- The currently selected cell deviates from the last executed cell and the user indicates that they want to work on a different analysis.
- You will be able to identify this based on the conversation history and the content of the currently selected and last executed cell.

**MOVE_TO_SPECIFIC**: User wants to add a cell at position that is specified in the prompt.
- The position may be identied based on a cell number, relative position, or content description.
- This will mostly involve a jump to a different analysis in the notebook, rather than a continuation of a current analysis by a single step.
- Note that the position of the currently selected cell may help in identifying this new position as the user might be looking at this section they want to move to.
""",

        "cell_selection_deletion_for_backtracking": """You select cells to delete during backtracking in a Jupyter notebook analysis workflow.

You are helping recover from a failed analysis by cleaning up cells that correspond to tasks that need to be redone. You will receive:
- **Reset tasks**: Tasks that were previously completed but are now set back to pending due to issues
- **Recovery objective**: Explanation of what went wrong and needs to be fixed

Identify which cells should be deleted to clean up the failed analysis attempts.
Delete cells that correspond to tasks that were reset.
Note that tasks were sequentially performed in the noteook, so cells mapping to tasks also appear sequentially, usually one cell per task.
Use the provided notebook structure to map cells to tasks:
make sure that you are certain about the identification of cells at boarders from kept to reset tasks,
e.g. make particularly sure that the first and last cell that you delete in a section both map to reset tasks 
and that the adjacent cells to these border cells (before first and after last if that exists) do not map to reset tasks.

Output:
Return "cells_to_delete" as a list of 0-based indices that correspond to cell numbers from the notebook structure (see notebook structure section).
""",

        "cell_selection_replacement": """
You select a cell number (target_cell) to replace code in within in a Jupyter notebook based on a user query.

Use 0-based indexing to select the cell number from the notebook structure (see notebook structure section).
Distinguish the following intents to guide the choice of target_cell.

**CONTINUE_LAST_EXECUTED**: User wants to work on the last executed cell
- The user wants to modify the last executed cell because there are errors or issues with the analysis.
- This is the default choice, only deviate from this choice if you see good evidence for one of the following intents.

**CONTINUE_CURSOR**: User wants to work on the currently selected cell
- The currently selected cell deviates from the last executed cell and the user indicates that they want to work on a new cell.
- You will be able to identify this based on the conversation history and the content of the currently selected and last executed cell.

**MOVE_TO_NEXT_FROM_LAST_EXECUTED**: User wants to work on the cell directly after the last executed cell.
- This mostly involves adapting the cell after the last executed cell to recent changes in the notebook.

**MOVE_TO_NEXT_FROM_CURSOR**: User wants to work on the cell directly after the selected cell.
- The currently selected cell deviates from the last executed cell and the user indicates that they do not just want to continue the notebook flow to the cell after the last executed cell.
- You will be able to identify this based on the conversation history and the content of the currently selected and last executed cell.

**MOVE_TO_SPECIFIC**: User wants to select a cell at position that is specified in the prompt.
- The position may be identied based on a cell number, relative position, or content description.
- This will mostly involve a jump to a different analysis in the notebook, rather than a continuation of a current analysis by a single step.
""",  

        "code_update": f"""Update the code in a jupyter notebook cell.

{PROMPT_BUILDING_BLOCKS['code_response']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
""",

        "code_generation": f"""Generate code for a jupyter notebook cell.

{PROMPT_BUILDING_BLOCKS['code_response']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
""",

        "code_generation_with_guidance": f"""Generate code for a jupyter notebook cell.

{PROMPT_BUILDING_BLOCKS['code_response']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
""",

        "reasoning_response_with_guidance": f"""Generate reasoning content for a jupyter notebook cell.

Reason about the provided problem/question based on intermediate analysis results.
Adhere to these rules:
- Output markdown-formatted text for a markdowncell in a jupyter notebook.
- Outline which analysis outputs in the notebook you are interpreting, summarizing what they show and interpret them with respect to the query.
  Make sure that you:
      * do not hallucinate any outputs to base your reasoning on.
      * interpret the relevant analysis results fully and not superficially or incompletely.
- Your reasoning will be interpreted by other agents - do not provide specific next steps, your role is to support them by providing reasoning on the problem/question you are given.
- Keep this reasoning concise and focussed on the problem/question at hand. View this as a link in a chain of thought over several tasks, do not introduce noise in this chain by providing overly lengthy or off-topic reasoning.
""",
        
        "code_review": """Review and explain code in context of the full analysis workflow.

Consider:
- Best practices from current bioinformatics documentation (knowledge base)
- Integration with existing session objects and workflow
- Current API usage and recommended patterns

Highlight any recommendations based on current standards.
""",

        "error_recovery": """An error occurred in code execution in a jupyter notebook cell.
Analyze the error and determine a recovery strategy from the following options:

**REPLACE_AND_RETRY**: Fix the code in the last executed cell that caused the error and execute it again.
- Choose this option if the error did not cause any non-recovarable change to key objects in the session:
For example, a subsetting of or a modification of gene expression features or medata of an anndata instance may not be recoverable.
A failed modification of metadata may be recoverable.
If uncertain and the notebook is relatively small / can be executed quickly, err on the side of caution and prefer REPLACE_AND_RESTART.
- Consider this a default choice.

**REPLACE_AND_RESTART**: Fix the code in the last executed cell that caused the error, restart the kernel and run all cells up to including the one that showed the error again.
- Choose this option if you find that REPLACE_AND_RETRY is not a suitable choice.
- This option is often favourable if you cannot trace or recover processing changes of the gene expression matrices in an anndata anymore.
""",

        "execution_monitor": """A cell has been executing for an extended period. Analyze the partial outputs and code to determine if execution should continue or be terminated.
Using the inputs, any intermediate outputs and your knowledge about the analyses in the cell, estimate a total run time for this cell.
If the estimated run time is low (e.g. below 10 minutes unless otherwise specified by the user) or higher but justified because the analysis cannot be replaced or accelerated, continue with the execution.
If the run time is high and not justified, terminate and suggest replacing the analysis by a similar one that is faster.
If you are highly uncertain about the total run time but suspect it might be very long, terminate and suggest a smaller trial version of this code or more stringent logging so that the cell can be updated and rerun with tighter run time control.
Be conservative - when in doubt, choose to continue.

Output the following fields:
- "action": "continue" or "terminate" to the currrently running cell
- "feedback": If you choose "terminate", suggest what needs to be changed.
""",

        "autoloop_intent_classification": f"""
You are analyzing user query during autonomous mode to determine the user's intent.

Classify their query as either:
1. TASK_LIST_MODIFICATION - User wants to modify the analysis strategy, i.e. the task list (add, remove, change task descriptions)
2. CODE_IMPLEMENTATION_FEEDBACK - User wants to change how specific tasks were implemented in code
3. APPROVAL - User agrees to current plan and wants the autonomous execution to continue.

Classify simple positive responses, such as "ok" or empty responses, as APPROVAL unless they contain other feedback.
""",

        "intent_classification": """Classify the user's request into one of these intents:

- **question_about_code**: User is asking about existing code, methods, or concepts
- **generate_code**: User wants to generate new code (will create new cells)  
- **generate_code_in_place**: User wants to modify/fix existing code (will replace current cell)
- **remove_code**: User wants to remove code

Consider the conversation context and determine the most appropriate intent.
Provide brief reasoning for your classification.
""",

        "question_answering": """Answer questions accurately using available sources.

Information source strategy:
- Knowledge base contains up-to-date bioinformatics APIs, workflows, and best practices
- Prioritize knowledge base for all bioinformatics-related information
- Only use additional sources if knowledge base lacks needed information

Clearly indicate which sources informed your answer.
""",

        "section_code_review": """You are reviewing a section of Jupyter notebook code that encountered an error during execution. 
Your goal is to fix the code to make this specific section run without errors.

## Context:
This is about **code maintenance and debugging**, not building new features. 
Focus on making the existing code work correctly with minimal changes.

You will receive:
- The complete code section being executed (all cells)
- The specific error that occurred and in which cell
- Any previous fix attempts in this section
- The conversation history for context about what this code is supposed to do

## Your Task:
Analyze the error and determine the minimal fix needed. You can:

**DELETE**: Remove cells that are causing issues or are no longer needed
- Use this for cells that are broken beyond simple fixes
- Use this for duplicate or conflicting cells
- Use this for cells that are no longer relevant

**REPLACE**: Fix existing code in place
- Use this for syntax errors, typos, or parameter issues
- Use this for updating deprecated function calls
- Use this for fixing variable names or imports

**INSERT**: Add new cells only if absolutely necessary
- Use this only when missing essential code (imports, setup)
- Keep additions minimal and focused on the specific error

## Important Guidelines:
- Focus on surgical fixes that resolve the immediate error with minimal disruption
- **Avoid creating conflicts** with analysis that may already exist later in the notebook
- Consider how your changes might affect downstream cells and variables
- Keep modifications minimal and targeted to the specific error
""",

        "task_list_generation": f"""Build a task list that addresses the user query.
Consider all rules and recommendations on task list design that are given in the following, make sure you do not ignore any of them.

{PROMPT_BUILDING_BLOCKS['task_list_description']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}

Output the following fields:
- "tasks": This is the task list. The value of this field is a list of tasks where each task is a dictonary with the fields:
            * "id" (integer): 1-based indexing in all tasks
            * "task" (string): description of task
            * "status": set to "pending" for all tasks
- "retrieval_queries": If the current reference workflow selection is not sufficient, you can supply "retrieval_queries" - a list of string queries that will then be used to find further reference workflow candidates that you can then use in the next step.
  This is typically applicable if ke tasks are not covered by reference workflows - are implemented as custom steps.
  If tasks are already adapted from reference workflows, but those references are not focussed on the analysis that you are adapting (for example because they use it as a small part of a larger analysis),
  query more focussed focussed reference workflows, e.g. tutorials dedicated to the analysis or tool you want to implement.
  For standard algorithms, for example commonly used unsupervised methods implemented in scanpy, search for general best practice or tutorial workflows that cover entire analyses rather than attempting to find specific tutorials.
  Aim for a 1-2 sentence summary of the content you are looking for per query, this query will be processed with a sentence embedding model and queried against a database.
  If and only if you think that the selection of reference workflows is sufficient to address the user query, you can skip this output.
""",

        "reference_workflow_selection": f"""You prepare a selection or reference workflows to guide developing an analysis plan for a user query.

Below, you are given a conversation history between an agent and the user with the instructions from the user.
You will prepare a selection of reference workflows that will help design analyses to address this query.
A reference workflow is a jupyter notebook that was published as an example/tutorial for a particular analysis concept, or as part of a reproducibility effort for published analysis results.

Your task now is to break the user's query down into groups of tasks that are covered by these reference workflows, for example:
    * the usage of dedicated python libraries that cover key parts of analyses requested by the user, 
    * specific applications, e.g. tissues, perturbations, diseases etc., 
    * or analysis objectives that may be addressed in more general tutorial notebooks.
You have two roles:
1) If you have access to summaries of "Putative reference workflows" below, identify workflows that cover the tasks in your task list.
   Consider these rules and guidelines:
    * DO NOT select workflows from the excluded workflows list (see section "Excluded workflows" if available) - these have been tried in previous iterations and returned no relevant content.
    * Importantly, your aim is not only just to select summaries that are maximally overlapping in language and objective to the entire user query,
      but also to find relevant reference workflows that are specific to subsets of tasks.
    * Try to cover each relevant task with a reference workflow, you can cover tasks with multiple alternative reference workflows if you do not exeed the limit:
      aim for returning up to 5 reference workflows in total.
    * If you are selecting a workflow to cover the usage of a specific python library and have access to multiple workflows that use this tool,
      prefer tutorial workflows that are dedicated to this tool, rather than workflows that use this tool as one of many in an application.
      Try to always provide at least one dedicated tutorial workflow for key analysis tools that have complex usage (e.g. have a complex API or come from dedicated python libraries).
    * For standard algorithms, for example commonly used unsupervised methods implemented in scanpy, prefer general best practice or tutorial workflows that cover entire analyses rather than attempting to find specific tutorials.
   Return your selection of reference workflows as "selected_notebooks".
2) Query summaries of further reference workflows that will be made accessible to you in the next iteration.
   Generate queries to find better reference workflows where current matches are suboptimal or missing.
   For example, query for dedicated tutorial workflows for specific tools or analysis methods to replace generic workflows.
   
Structure your output as follows:
- "selected_notebooks": a list with the Notebook IDs that you choose (use the full path shown in "Notebook ID" field, e.g., "scverse/scanpy-tutorials/pbmc3k.ipynb").
- "retrieval_queries": a list of retrieval queries that will be used to extend and refine the list of summaries in subsequent iterations.
    Aim for a 1-2 sentence semantic description of the content you are looking for per query. Focus on concepts, methods, and biological context.
    Each query will be processed with a sentence embedding model and matched against workflow summaries using semantic similarity.
    Only return an empty list if you are highly confident the current selection comprehensively addresses all relevant tasks.
""",

        "reference_workflow_selection_only": f"""You are adding new reference workflows based on retrieval queries to an existing selection.

Below you will find:
    * Currently selected reference workflows (already in use)
    * Retrieval queries (new topics to find workflows for)
    * Putative reference workflows (candidates to choose from)
    * Excluded workflows (workflows that returned no relevant content in past iterations - DO NOT select these)
    * Current task list draft that addresses the user's query
You will extend the selection of reference workflows that will help improve that task list.
A reference workflow is a jupyter notebook that was published as an example/tutorial for a particular analysis concept, or as part of a reproducibility effort for published analysis results.
Your task now to generate an updated version of the current list of reference workflow IDs.
Start with the current list of selected reference workflow IDs that you can find below.
Consider these rules and guidelines for adding and removing reference workflows based on this current selection:
    * DO NOT REMOVE workflows that are currently cited in tasks of the task list.
    * DO NOT ADD workflows from the excluded workflows list (see section "Excluded workflows" if available) - these have been tried in previous iterations and returned no relevant content.
    * The retrieval queries correspond to concepts that the agent that buit the task list wanted to dive deeper into 
      but wan't able to find good references for in the current list of reference workflows.
      Use these retrieval queries to select workflows to ADD so that they are available for the next update of the task list.
      These retrieval queries should be the major selection determinant for ADDING new workflows.
      Often, these queries serve distinct analysis aims and would therefore be best addressed through distinct, specialized workflows that address individual queries, 
      rather than worlflows that loosely match overall thems of the task list.
    * As a secondary selection criterion for ADDING workflows, you can consider tasks from the task list that are currently not citing reference workflows.
    * If you are selecting a workflow to cover the usage of a specific python library and have access to multiple workflows that use this tool,
      prefer tutorial workflows that are dedicated to this tool, rather than workflows that use this tool as one of many in an application.
      Try to always provide at least one dedicated tutorial workflow for key analysis tools that have complex usage (e.g. have a complex API or come from dedicated python libraries).
    * For standard algorithms, for example commonly used unsupervised methods implemented in scanpy, prefer general best practice or tutorial workflows that cover entire analyses rather than attempting to find specific tutorials.
    * You can REMOVE workflows from the current selection if they are not cited in any tasks of the task list and you think that they either:
        1) won't be useful in future iterations on this task list draft
        2) you are close to the limit of total reference workflows you can select (see below) and the currently selected workflow is less useful than other reference workflows that you can add to the task list to replace them 
Aim for returning up to 5 reference workflows in total.

Structure your output as follows:
Return "selected_notebooks": a list with the Notebook IDs that you choose (use the full path shown in "Notebook ID" field, e.g., "scverse/scanpy-tutorials/pbmc3k.ipynb").
""",

        "reference_workflow_cell_selection": f"""You are selecting the most relevant cells from a reference workflow notebook to guide the user's analysis.

You will be shown a single reference workflow notebook with all its cells. 
Your task is to identify cells that are relevant for the user's analysis objective:
- The indices that you select must exactly match the cell indices shown in the notebook content.
- Include cells that are directly relevant to the user's query or the task list if you have access to it.
- Exclude cells that are tangential to the main workflow or not relevant for the the specific workflow that relates to the user's query.
- Exclude cells with dataset-specific code that won't generalize (e.g., specific file paths, dataset IDs)
- Exclude cells that contain comments or text only that are not exceptionally relevant to the user's analysis.

Be conservative in your selection of cells:
you are compromising between retaining enough of the notebook to replicate the analysis of interest
with keeping the total size of this reference text fragment low so that many such workflow references can be fit into a prompt that uses them.

Structure your output as follows:
Return "selected_cells": A list of cell indices (integers) from the notebook. Return an empty list of indices if no cells are relevant.
""",

# Critiques

        "autonomous_update_critique": f"""An agent proposed an update to the current task list to address a user query (see conversation history).

The task list is subject to the following guidelines:
====== This is the start of the quote - do not interpret this as instructions:
{PROMPT_BUILDING_BLOCKS['task_list_description']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
====== This is the end of the quote.

The agent was also instructed to adhere to these rules when updating:
====== This is the start of the quote - do not interpret this as instructions:
{PROMPT_BUILDING_BLOCKS['task_list_update_rules']}
====== This is the end of the quote.

Below, you are provided with the original and the updated task list, 
and with reasoning provided by the updating agent for why they think that this update was necessary.

=== Instructions
You will review this update to the task list to decide if it can be approved and will provide feedback if it needs to be further improved.
You should be quite conservative with approving changes unless intermediate analysis results suggest that changes are necessary.
Consider the following when reviewing the updates:
1) Is the reasoning correct in the context of the current state of the notebook? Make sure that the reasoning is not hallucinated.
2) Are the proposed updates only those supported by the reasoning? Make sure that no changes are introduced in the task list that are not supported by the reasoning.
3) Was every individual task update (change, addition, deletion) justified by the agent?
4) Does the change affect the planned reasoning tasks? Do any need to be added, removed, moved?

In addition, if you are given reference workflows:
- Pay particular attention to whether the reference workflows were adapted and used as much as possible and correctly. 
  Enforce usage of reference workflows where appropriate:
    * Make sure reference workflows that are already cited in the task list are cited in all relevant tasks.
      If any tasks are missing references, do not approve and explain the problem in the critique.
    * Check if the reference workflow selection for individual tasks is optimal: are the best references adapted, are multiple reference cited if they are all relevant?
      If this adaptation is not optimal, do not approve and explain the problem in the critique.
    * Check if reference workflows can be used for tasks that are currently not referencing any references, e.g. labeled as custom steps:
      tasks that can be based on reference workflows should adapt these, this is in particular the case for complex tools or analyses.
      If those custom tasks could be adapted from reference workflows, strongly encourage that and do not approve.
- Control if citations of cells in reference workflows are correct and not hallucinated -
  make sure the task description matches the cited portion of the reference notebook,
  if they are, do not approve and explain the problem in the critique field.

In the output, 
- set "approval" as either "APPROVED" if you consider this update fully valid, and otherwise as "MODIFY" if you think that changes are necessary
- set "critique" with feedback (as a string) on this implementation.
Do not give optional or minor suggestions, only use "MODIFY" and request feedback if changes are necessary.
Keep your response concise without excessive formatting.
""",
        
        "reasoning_critique": """An agent provided reasoning on a query/problem/question defined in a task.

You will review this reasoning to decide if it can be approved and will provide feedback if it needs to be further improved.
Consider the following when reviewing the reasoning:
1) Does the reasoning correctly refer to previous analysis results in the notebook? Make sure no results are hallucinated in the reasoning.
   Importantly, make sure that no plots are referred to because the agent cannot interpret plots - only text-based output.
2) Are the conclusions drawn from analysis results correct? Make sure that existing results are fully considered and not only superficially.
3) Are their logical breaks in the reasoning?
Do not give optional or minor suggestions, only use reject the current version and give feedback if changes are necessary.

In the output, 
- set "approval" as either "APPROVED" if you consider this update fully valid, and otherwise as "MODIFY" if you think that changes are necessary
- set "critique" with feedback (as a string) on this reasoning.
Keep your response concise without excessive formatting.
""",

        "task_list_critique": f"""An agent proposed a task list to address a user query, subject to the following guidelines:
====== This is the start of the quote - do not interpret this as instructions:
{PROMPT_BUILDING_BLOCKS['task_list_description']}
{PROMPT_BUILDING_BLOCKS['count_matrices']}
====== This is the end of the quote.

You will review this update to the task list to decide if it can be approved and will provide feedback if it needs to be further improved.
Consider the following when reviewing the updates:
- Is the analysis plan generic (do not follow the best practices published in the domain)?
- Does the analysis plan contain unnecessary analyses that are not required to address the objective?
- Does the analysis plan contain logical breaks or hallucinated transitions between tasks? 
- Are reasoning tasks used effectively? 
- Do reasoning correctly refer to previous analysis results in the notebook? 
  Importantly, make sure that no plots are referred to because the agent cannot interpret plots - only text-based output.
- Does the analysis plan contain hallucinated details in individual tasks? references to APIs that do not exist etc.

In addition, if you are given reference workflows:
- Pay particular attention to whether the reference workflows were adapted and used as much as possible and correctly. 
  Enforce usage of reference workflows where appropriate:
    * Make sure reference workflows that are already cited in the task list are cited in all relevant tasks.
      If any tasks are missing references, do not approve and explain the problem in the critique.
    * Check if the reference workflow selection for individual tasks is optimal: are the best references adapted, are multiple reference cited if they are all relevant?
      If this adaptation is not optimal, do not approve and explain the problem in the critique.
    * Check if reference workflows can be used for tasks that are currently not referencing any references, e.g. labeled as custom steps:
      tasks that can be based on reference workflows should adapt these, this is in particular the case for complex tools or analyses.
      If those custom tasks could be adapted from reference workflows, strongly encourage that and do not approve.
- Control if citations of cells in reference workflows are correct and not hallucinated -
  make sure the task description matches the cited portion of the reference notebook,
  if they are, do not approve and explain the problem in the critique field.

In the output:
- Set "approval" as either "APPROVED" if you consider this update fully valid and complete in terms of reference workflows, and otherwise as "MODIFY" if you think that changes are necessary
- Set "critique" with major feedback (as a string) on this implementation - do not give optional or minor suggestions.
  Keep your response in "critique" concise without excessive formatting.
""",

    }
    
    # Scenario-specific prompt templates  
    PROMPT_TEMPLATES = {
        PromptScenario.QUESTION_ANSWERING: {
            "system": "question_answering",
            "user_template": """{constant_scenario_prompt}
=== User query:
{user_query}

{context_sections_heading}
{context_sections_rag}
{context_sections_conversation_history}
{context_sections_reference_workflow_content}

{rag_section}
{conversation_history_section}
{reference_workflow_section}""" 
        },

        PromptScenario.REFERENCE_WORKFLOW_SELECTION: {
            "system": "reference_workflow_selection", 
            "user_template": """{constant_scenario_prompt}
{retrieval_query_section}
{reference_workflow_ids_section}
{conversation_history_section}

{reference_workflow_preselection_section}"""
        },

        PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY: {
            "system": "reference_workflow_selection_only",
            "user_template": """{constant_scenario_prompt}
{retrieval_query_section}
{reference_workflow_ids_section}
{task_list_section}
{conversation_history_section}
{reference_workflow_preselection_section}
{reference_workflow_section}"""
        },

        PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION: {
            "system": "reference_workflow_cell_selection",
            "user_template": """{constant_scenario_prompt}
{retrieval_query_section}

=== User query:
{user_query}
{task_list_section}

{conversation_history_section}

=== Current notebook for cell selection:
{current_notebook_cell_selection_section}"""
        },

        PromptScenario.CODE_GENERATION: {
            "system": "code_generation", 
            "user_template": """{constant_scenario_prompt}
=== User query:
{user_query}

{context_sections_heading}
{context_sections_rag}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_notebook_structure}

{rag_section}
{conversation_history_section}
{execution_history_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.CODE_GENERATION_WITH_GUIDANCE: {
            "system": "code_generation_with_guidance",
            "user_template": """{constant_scenario_prompt}

This code generation is guided by a specific task objective from an active task list.
Adhere to these rules:
- Focus on implementing the specific task objective provided
- Generate code that directly addresses the task goal
- Consider execution context and previous results from the notebook
- If you are given citations of positions in reference workflows to adapt as part of the guidance,
  focus the scope of the code generation on these positions in reference notebooks and try to reflect this code as closely as reasonable.
  Try to implement the cited code cells from the references as directly as possible, adhering to the following rules and guidelines.
  Often, the more similar you code is to the code presented in these reference cells, the more transparent your analysis will be.

{active_vs_next}

=== User query:
{user_query}

{context_sections_heading}
{context_sections_rag}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_notebook_structure}

{rag_section}
{conversation_history_section}
{execution_history_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.CODE_FIXING: {
            "system": "code_update",
            "user_template": """{constant_scenario_prompt}

Follow these specific rules:
- Update the code to fix the error in the cell that you are given.
- Do not remove code from the cell that is required for the cell to run:
  for example, do not remove a declaration that defines a variable that your proposed new code depends on.
=== Code of cell that caused the error:
```
{current_cell}
```
If you need to understand the context of this cell in the notebook to fix the issue, 
you can find the content of the entire jupyter notebook in the session context notebook structure section.
If retrieval is enabled, you are provided a retrieval section with snippets relevant to the specific issue, 
and a reference workflow section that is used to guide the code in this entire notebook.
You can use both as appropriate to fix errors, including overall API usage, keyword definitions and defaults, best practices, etc.

{error_section}

{context_sections_heading}
{context_sections_execution_history}
{context_sections_rag}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{execution_history_section}
{rag_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.CODE_FIXING_WITH_GUIDANCE: {
            "system": "code_update",
            "user_template": """{constant_scenario_prompt}

Follow these specific rules:
- Update the code to fix the error in the cell that you are given.
- Note that this cell was intended to address the following objective, make sure that the updated code addresses it:

{active_task_objective}

- If you are given citations of positions in reference workflows to adapt as part of the guidance, 
  focus the scope of the code generation on these positions in reference notebooks and try to reflect this code as closely as reasonable.
  Try to implement the cited code cells from the references as directly as possible, adhering to the following rules and guidelines.
  Often, the more similar you code is to the code presented in these reference cells, the more transparent your analysis will be.
- Do not remove code from the cell that is required for the cell to run:
  for example, do not remove a declaration that defines a variable that your proposed new code depends on.

{retry_objective_section}

=== Code of cell that caused the issue:
```
{current_cell}
```
{error_section}

{context_sections_heading}
{context_sections_execution_history}
{context_sections_rag}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{execution_history_section}
{rag_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.CODE_UPDATE_WITH_GUIDANCE: {
            "system": "code_update",
            "user_template": """{constant_scenario_prompt}

Follow these specific rules:
- Note that this cell was intended to address the following objective, make sure that the updated code addresses it:

{active_task_objective}

- If you are given citations of positions in reference workflows to adapt as part of the guidance, 
  focus the scope of the code generation on these positions in reference notebooks and try to reflect this code as closely as reasonable.
  Try to implement the cited code cells from the references as directly as possible, adhering to the following rules and guidelines.
  Often, the more similar you code is to the code presented in these reference cells, the more transparent your analysis will be.
- Do not remove code from the cell that is required for the cell to run:
  for example, do not remove a declaration that defines a variable that your proposed new code depends on.

{retry_objective_section}

=== Code of cell to update:
```
{current_cell}
```
{error_section}

{context_sections_heading}
{context_sections_execution_history}
{context_sections_rag}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{execution_history_section}
{rag_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.REASONING_RESPONSE_WITH_GUIDANCE: {
            "system": "reasoning_response_with_guidance",
            "user_template": """{constant_scenario_prompt}
The problem/question you need to reason about is:
{active_task_objective}
{reasoning_instructions_section}

{context_sections_heading}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.REASONING_CRITIQUE: {
            "system": "reasoning_critique",
            "user_template": """{constant_scenario_prompt}
The problem/question that was reasoned about is:
{active_task_objective}
{reasoning_critique_instructions_section}

{context_sections_heading}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{notebook_structure_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.CODE_REVIEW: {
            "system": "code_review",
            "user_template": """{constant_scenario_prompt}
=== Original user query:
{user_query}

=== Current code:
```
{current_cell}
```

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{conversation_history_section}
{execution_history_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.CELL_SELECTION_ADDITION: {
            "system": "cell_selection_addition",
            "user_template": """{constant_scenario_prompt}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_notebook_structure}

{conversation_history_section}
{execution_history_section}
{notebook_structure_section}"""
        },
        
        PromptScenario.CELL_SELECTION_DELETION_FOR_BACKTRACKING: {
            "system": "cell_selection_deletion_for_backtracking",
            "user_template": """{constant_scenario_prompt}

{cell_selection_deletion_section}
            
{context_sections_heading}
{context_sections_execution_history}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{execution_history_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.CELL_SELECTION_REPLACEMENT: {
            "system": "cell_selection_replacement",
            "user_template": """{constant_scenario_prompt}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_notebook_structure}

{conversation_history_section}
{execution_history_section}
{notebook_structure_section}"""
        },
        
        PromptScenario.AUTONOMOUS_MARK_COMPLETION: {
            "system": "autonomous_mark_completion",
            "user_template": """{constant_scenario_prompt}
{task_list_section}

{error_section}

{context_sections_heading}
{context_sections_execution_history}
{context_sections_reference_workflow_content}

{execution_history_section}
{reference_workflow_section}"""
        },
        
        PromptScenario.AUTONOMOUS_UPDATE_TASKS: {
            "system": "autonomous_update_tasks",
            "user_template": """{constant_scenario_prompt}
{task_list_update_instructions}
{error_section}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_rag}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{conversation_history_section}
{execution_history_section}
{rag_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.AUTONOMOUS_UPDATE_CRITIQUE: {
            "system": "autonomous_update_critique",
            "user_template": """{constant_scenario_prompt}
{task_list_update_critique_instructions}
{error_section}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_execution_history}
{context_sections_rag}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{conversation_history_section}
{execution_history_section}
{rag_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.BACKTRACK_RECOVERY: {
            "system": "backtrack_recovery",
            "user_template": """{constant_scenario_prompt}

=== Code of cell that caused the error:
```
{current_cell}
```

{error_section}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_notebook_structure}

{conversation_history_section}
{notebook_structure_section}"""
        },

        PromptScenario.ERROR_RECOVERY: {
            "system": "error_recovery",
            "user_template": """{constant_scenario_prompt}

=== Code of cell that caused the error:
```
{current_cell}
```

{error_section}

{task_list_section}

{context_sections_heading}
{context_sections_notebook_structure}

{notebook_structure_section}"""
        },

        PromptScenario.EXECUTION_MONITOR: {
            "system": "execution_monitor",
            "user_template": """{constant_scenario_prompt}

{execution_monitor_section}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_notebook_structure}
{context_sections_reference_workflow_content}

{conversation_history_section}
{notebook_structure_section}
{reference_workflow_section}"""
        },

        PromptScenario.TASK_LIST_GENERATION: {
            "system": "task_list_generation",
            "user_template": """=== Instructions
{constant_scenario_prompt}
{task_list_generation_instructions}
=== The user query that motivated this task list was:
{user_query}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_reference_workflow_content}
{context_sections_notebook_structure}

{conversation_history_section}
{reference_workflow_section}
{notebook_structure_section}"""
        },

        PromptScenario.TASK_LIST_CRITIQUE: {
            "system": "task_list_critique",
            "user_template": """{constant_scenario_prompt}
{task_list_generation_critique_instructions}
=== The user query that motivated this task list was:
{user_query}

{context_sections_heading}
{context_sections_conversation_history}
{context_sections_reference_workflow_content}
{context_sections_notebook_structure}

{conversation_history_section}
{reference_workflow_section}
{notebook_structure_section}"""
        },
        
        PromptScenario.INTENT_CLASSIFICATION: {
            "system": "intent_classification",
            "user_template": """{constant_scenario_prompt}

=== User query:
{user_query}

{context_sections_heading}
{context_sections_conversation_history}

{conversation_history_section}"""
        },
        
        PromptScenario.AUTOLOOP_INTENT_CLASSIFICATION: {
            "system": "autoloop_intent_classification",
            "user_template": """{constant_scenario_prompt}

=== User query:
{user_query}

{task_list_section}

{context_sections_heading}
{context_sections_conversation_history}

{conversation_history_section}"""
        }
    }
    
    def generate_prompt(self, exec_context: 'ExecutionContext', scenario: PromptScenario, model_name: str = "", structured_output: bool = True, reasoning_level: Optional[str] = None) -> tuple[str, str]:
        """Generate system and user prompts for the given execution exec_context.

        Args:
            exec_context: ExecutionContext with all necessary data
            scenario: The prompt scenario to use
            model_name: The LLM model name to generate appropriate prompts for
            structured_output: If False, append JSON format instructions to system prompt
            reasoning_level: Reasoning level for OSS models (low/medium/high)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self.PROMPT_TEMPLATES[scenario]

        # Build system prompt: base + scenario-specific
        # Use agentic prompt if it's an agentic model, otherwise use shared prompt
        system_prompt = self.get_system_prompt(model_name, reasoning_level=reasoning_level)
        scenario_key = template["system"]
        constant_scenario_prompt = self.SCENARIO_PROMPTS[scenario_key]
        
        # Build user prompt with conditional sections
        user_template = template["user_template"]
        
        # Build contextual sections (only build RAG section if template uses it)
        sections = {
            "constant_scenario_prompt": constant_scenario_prompt,
            "execution_history_section": self._build_execution_history_section(exec_context),
            "execution_monitor_section": self._build_execution_monitor_section(exec_context),
            "conversation_history_section": self._build_conversation_history_section(exec_context),
            "notebook_structure_section": self._build_notebook_structure_section(exec_context, scenario),
            "error_section": self._build_error_section(exec_context),
            # Optional sections:
            "active_vs_next": self._build_active_vs_next_section(exec_context),
            "cell_selection_deletion_section": self._build_cell_selection_deletion_section(exec_context),
            "rag_section": self._build_rag_section(exec_context),
            "reasoning_instructions_section": self._build_reasoning_instructions_section(exec_context),
            "reasoning_critique_instructions_section": self._build_reasoning_critique_instructions_section(exec_context),
            "reference_workflow_preselection_section": self._build_reference_workflow_preselection_section(exec_context),
            "reference_workflow_section": self._build_reference_workflow_section(exec_context),
            "task_list_section": self._build_task_list_section(exec_context),
            "task_list_update_instructions": self._build_task_list_update_instructions_section(exec_context),
            "task_list_update_critique_instructions": self._build_task_list_update_critique_instructions_section(exec_context),
            "task_list_generation_instructions": self._build_task_list_generation_instructions_section(exec_context),
            "task_list_generation_critique_instructions": self._build_task_list_generation_critique_instructions_section(exec_context),
            "retrieval_query_section": self._build_retrieval_query_section(exec_context),
            "reference_workflow_ids_section": self._build_reference_workflow_ids_section(exec_context),
            "retry_objective_section": self._build_retry_objective_section(exec_context),
            "current_notebook_cell_selection_section": self._build_current_notebook_cell_selection_section(exec_context),
            # Directly from context
            "active_task_objective": exec_context.inputs.context.get("active_task_objective", ""),
            "retry_objective": exec_context.inputs.context.get("retry_objective", ""),
        }
        
        # Fill in template
        user_prompt = user_template.format(
            user_query=exec_context.inputs.user_query,
            current_cell=exec_context.inputs.context['current_cell'],
            **sections, **self.PROMPT_SECTION_SUMMARIES
        )

        # Add JSON format instructions if structured output is disabled
        if not structured_output:
            json_instruction = self._get_json_format_instruction(scenario)
            if json_instruction:
                system_prompt = f"{system_prompt}\n\n{json_instruction}"

        return system_prompt, user_prompt
              
    def _build_active_vs_next_section(self, exec_context: 'ExecutionContext') -> str:
        """Build task list context section from input."""
        if "active_task_objective" in exec_context.inputs.context.keys():
            task_text = f"Generate code for this task from the task list:\n {exec_context.inputs.context['active_task_objective']}\n"
            if "next_pending_task_objective" in exec_context.inputs.context.keys():
                task_text += f"\nAvoid overlaps with the subsequent task in the list:\n\n{exec_context.inputs.context['next_pending_task_objective']}\n\n"
            task_text += "You can find the full task list in the conversation history section."  
        else:
            return ""
        return task_text
    
    def _build_cell_selection_deletion_section(self, exec_context: 'ExecutionContext') -> str:
        reset_tasks = exec_context.inputs.context.get("reset_tasks", None)
        backtracking_context = exec_context.inputs.backtracking_context

        task_text = []
        if reset_tasks:
            task_text.append("Tasks Reset to Pending (need cleanup):")
            for task in reset_tasks:
                task_desc = task["task"]
                task_text.append(f"- **Task {task['id']}**: {task_desc}")
            task_text.append("")

        if backtracking_context and backtracking_context.is_active:
            task_text.append(f"Recovery Objective:\n{backtracking_context.recovery_objective}")
            task_text.append("")
        task_text = "\n".join(task_text)
        return task_text

    def _build_reasoning_critique_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        """This section is made available for reasoning critique."""
        if "reasoning_response" in exec_context.inputs.context.keys():
            previous_reasoning = exec_context.inputs.context["reasoning_response"]
            # Handle None values
            if previous_reasoning is None:
                previous_reasoning = "(No previous reasoning)"

            task_text = [
                "\n====== This is the proposed reasoning:",
                previous_reasoning,
                "====== This is the end of the proposed reasoning.",
            ]
            if "reasoning_critique" in exec_context.inputs.context.keys():
                reasoning_critique = exec_context.inputs.context["reasoning_critique"]
                # Handle None values
                if reasoning_critique is None:
                    reasoning_critique = "(No critique)"

                task_text.extend([
                    "\nThis proposed reasoning is an update that was based on your feedback the last iteration:",
                    "====== This was your feedback in the previous iteration that led to this new reasoning:",
                    reasoning_critique,
                    "====== This is the end of your previous feedback.",
                ])
            task_text.extend([""])
            task_text = "\n".join(task_text)
            return task_text
        else:
            return ""

    def _build_reasoning_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        """This section is made available for reasoning based on a critique (after the first iteration)."""
        if ("reasoning_critique" in exec_context.inputs.context.keys() and
            "reasoning_response" in exec_context.inputs.context.keys()):
            previous_reasoning = exec_context.inputs.context["reasoning_response"]
            reasoning_critique = exec_context.inputs.context["reasoning_critique"]

            # Handle None values
            if previous_reasoning is None:
                previous_reasoning = "(No previous reasoning)"
            if reasoning_critique is None:
                reasoning_critique = "(No critique)"

            task_text = [
                "\n====== This is the previous reasoning:",
                previous_reasoning,
                "====== This is the end of your previous reasoning.",
                "\nThis reasoning was critiqued as follows - adapt your reasoning if necessary.",
                "Only change this existing reasoning according to what was critiqued or what was wrong.",
                "====== This is the feedback:",
                reasoning_critique,
                "====== This is the end of the feedback.",
            ]
            task_text.extend([""])
            task_text = "\n".join(task_text)
            return task_text
        else:
            return ""
    
    def _build_task_list_section(self, exec_context: 'ExecutionContext') -> str:
        """Build task list context section from input."""
        section_heading = "=== Current task list:\n"
        if exec_context.inputs.task_list is not None and "tasks" in exec_context.inputs.task_list:
            task_text = format_task_list(exec_context.inputs.task_list)
        else:
            return ""
        return section_heading + task_text

    def _build_task_list_generation_critique_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        """This section is made available for critiques in initial planning."""
        task_text = []
        if "task_text_old" in exec_context.inputs.context.keys():
            task_text_old = exec_context.inputs.context["task_text_old"]
            # Handle None values
            if task_text_old is None:
                task_text_old = "(No previous task list)"
            task_text.extend([
                "\nThis is the previous version of the task list:\n",
                task_text_old
            ])
        task_text_new = format_task_list(exec_context.inputs.task_list)
        task_text.extend([
            "\nThis is the current version of the task list:\n",
            task_text_new,
        ])
        if "task_list_critique" in exec_context.inputs.context.keys():
            task_list_critique = exec_context.inputs.context["task_list_critique"]
            # Handle None values
            if task_list_critique is None:
                task_list_critique = "(No critique)"
            task_text.extend([
                "\nYou provided the following feedback on the previous version that led to this update:\n",
                task_list_critique,
            ])
        task_text.extend([""])
        task_text = "\n".join(task_text)
        return task_text
        
    def _build_task_list_generation_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        """This section is made available for initial planning if a critique was given (after the first iteration)."""
        if "task_list_critique" in exec_context.inputs.context.keys():
            task_text_new = format_task_list(exec_context.inputs.task_list)
            task_list_critique = exec_context.inputs.context["task_list_critique"]
            # Handle None values
            if task_list_critique is None:
                task_list_critique = "(No critique)"
            task_text = [
                "\nThis is the current version of the task list:\n",
                task_text_new,
                "\nThe agent provided the following feedback for this update:\n",
                task_list_critique,
                "\nImprove the current version of the task list based on this feedback.",
            ]
        else:
            task_text = [""]
        task_text = "\n".join(task_text)
        return task_text

    def _build_task_list_update_critique_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        """This section is made available for initial planning if a critique was given (after the first iteration)."""
        if "task_list_update_rationale" in exec_context.inputs.context.keys():
            # If task_list_update_rationale is set, task_text_old is also given
            # because the update was performed on an existing task list.
            task_text_old = exec_context.inputs.context["task_text_old"]
            task_text_new = format_task_list(exec_context.inputs.task_list)
            task_list_update_rationale = exec_context.inputs.context["task_list_update_rationale"]

            # Handle None values
            if task_text_old is None:
                task_text_old = "(No previous task list)"
            if task_list_update_rationale is None:
                task_list_update_rationale = "(No rationale)"

            task_text = [
                "\nThis is the original task list:\n",
                task_text_old,
                "\nThis is the draft of the updated task list:\n",
                task_text_new,
                "\nThe agent provided the following reasoning for modifying the task list:\n",
                task_list_update_rationale,
            ]
            if "autonomous_update_critique" in exec_context.inputs.context.keys():
                autonomous_update_critique = exec_context.inputs.context["autonomous_update_critique"]
                # Handle None values
                if autonomous_update_critique is None:
                    autonomous_update_critique = "(No critique)"
                task_text.extend([
                    "\nYou provided the following feedback on the previous version that led to this update:\n",
                    autonomous_update_critique,
                ])
            task_text.extend([""])
            task_text = "\n".join(task_text)
            return task_text
        else:
            return ""
        
    def _build_task_list_update_instructions_section(self, exec_context: 'ExecutionContext') -> str:
        if "autonomous_update_critique" in exec_context.inputs.context.keys():
            # If autonomous_update_critique is set, task_text_old is also given
            # because the critique was performed on an update to an existing task list.
            task_text_old = exec_context.inputs.context["task_text_old"]
            task_text_new = format_task_list(exec_context.inputs.task_list)
            autonomous_update_critique = exec_context.inputs.context["autonomous_update_critique"]

            # Handle None values
            if task_text_old is None:
                task_text_old = "(No previous task list)"
            if autonomous_update_critique is None:
                autonomous_update_critique = "(No critique)"

            task_text = "\n".join([
                "\nThis is the original task list:\n",
                task_text_old,
                "\nThis is the draft of the updated task list:\n",
                task_text_new,
                "\nThe agent provided the following feedback for this update:\n",
                "====== This is the start of the quote from the agent:",
                autonomous_update_critique,
                "====== This is the end of the quote from the agent.",
                "",
                self.PROMPT_BUILDING_BLOCKS['task_list_update_critiqued'],
                "",
                self.PROMPT_BUILDING_BLOCKS['task_list_update_output_critiqued'],
                "",
            ])
            return task_text
        else:
            # This is the section if no critique is given - ie the standard first pass.
            task_text = format_task_list(exec_context.inputs.task_list)
            task_text = "\n".join([
                "\nThis is the current task list:\n",
                task_text,
                "",
                self.PROMPT_BUILDING_BLOCKS['task_list_update'],
                "",
                self.PROMPT_BUILDING_BLOCKS['task_list_update_output'],
                "",
            ])
            return task_text
            
    def _build_reference_workflow_section(self, exec_context: 'ExecutionContext') -> str:
        """Build reference workflow context section with header list."""
        if "reference_workflow_content" not in exec_context.inputs.context:
            return ""

        content_dict = exec_context.inputs.context["reference_workflow_content"]
        if not content_dict:
            return ""

        # Get full IDs for header - need to extract from content sections
        # Each content value starts with "> Notebook ID: full_id"
        full_ids = []
        for content in content_dict.values():
            # Extract full ID from first line of content
            first_line = content.split('\n')[0] if content else ""
            if first_line.startswith("> Notebook ID:"):
                full_id = first_line.replace("> Notebook ID:", "").strip()
                full_ids.append(full_id)

        # Build header with list
        header_lines = [
            "\n=== Session context: Reference workflows",
            "Sections for new notebooks are prefixed with '>', cells within notebooks with '>>'.",
            "This section has content of the following notebooks in this order:",
        ]
        for full_id in full_ids:
            header_lines.append(f"- Notebook ID: {full_id}")
        header_lines.append("")

        # Convert dict to string by concatenating all workflow sections
        rag_text = "\n\n".join(content_dict.values())

        return "\n".join(header_lines) + rag_text

    def _build_reference_workflow_preselection_section(self, exec_context: 'ExecutionContext') -> str:
        """Build reference workflow preselection context section from retrieval tool results."""
        sections = []

        # Add excluded workflows (workflows that returned empty indices)
        if exec_context.inputs.excluded_workflows:
            excluded_list = "\n".join([f"- {wf}" for wf in exec_context.inputs.excluded_workflows])
            sections.append("\n=== Session context: Excluded workflows")
            sections.append("These have been counterselected before, do not select these:\n" + excluded_list)

        # Add putative workflows
        if "putative_reference_workflow_summaries" in exec_context.inputs.context:
            rag_text = exec_context.inputs.context["putative_reference_workflow_summaries"]
            sections.append("\n=== Session context: Putative reference workflows\n" + rag_text)

        return "\n".join(sections) if sections else ""
        
    def _build_current_notebook_cell_selection_section(self, exec_context: 'ExecutionContext') -> str:
        """Build section showing the current notebook for cell selection."""
        if "current_notebook_for_cell_selection" not in exec_context.inputs.context:
            return ""

        notebook_info = exec_context.inputs.context["current_notebook_for_cell_selection"]
        notebook_id = notebook_info.get("notebook_id", "unknown")
        notebook_data = notebook_info.get("notebook_data", {})

        # Format the notebook content similar to _format_notebook_content
        metadata = notebook_data.get("metadata", {})
        full_notebook_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"

        content_parts = []
        content_parts.append(f"Notebook ID: {full_notebook_id}")
        content_parts.append(f"Title: {metadata.get('title', notebook_id)}")
        content_parts.append("")

        cells = notebook_data.get("cells", [])
        current_section = None
        for cell in cells:
            content = cell.get("content", "")
            idx = cell.get("order", "")
            section = cell.get("section", "")

            if section and section != current_section and section != "main":
                content_parts.append(f"\n## Section: {section}\n")
                current_section = section

            content_parts.append(f"Cell {idx}:")
            content_parts.append(content)
            content_parts.append("")

        return "\n".join(content_parts)

    def _build_reference_workflow_ids_section(self, exec_context: 'ExecutionContext') -> str:
        if "reference_workflow_ids" not in exec_context.inputs.context:
            return ""

        reference_workflow_ids = exec_context.inputs.context["reference_workflow_ids"]

        # Don't show section if IDs are empty
        if not reference_workflow_ids or reference_workflow_ids.strip() == "":
            return ""

        section_heading = "\nCurrent selection of reference workflow IDs:"
        return section_heading + "\n" + reference_workflow_ids

    def _build_retrieval_query_section(self, exec_context: 'ExecutionContext') -> str:
        """Build retrieval query section."""
        if "retrieval_queries" in exec_context.inputs.context and \
            exec_context.inputs.context["retrieval_queries"] is not None and \
            len(exec_context.inputs.context["retrieval_queries"]) > 0:
            retrieval_queries = exec_context.inputs.context["retrieval_queries"]
            retrieval_text = "\n".join([
                "An agent highlighted the following topics as being of interest for improving the task list - consider these in particular when selecting new/additional reference workflows:\n",
            ] + retrieval_queries + [""])
            return retrieval_text
        else:
            return ""
            
    def _build_rag_section(self, exec_context: 'ExecutionContext') -> str:
        """Build RAG context section from RAG retrieval tool results."""
        section_heading = "\n".join([
            "\n=== Session context: Scenario-specific retrieval",
            "These are snippets of API documentation or example/tutorial/reproducibility workflows.",
        ])

        # This retrieval is optional - return empty string if not executed
        if "rag_retrieval" not in exec_context.inputs.context:
            return ""

        # RAG retrieval tool returns a string directly via results["content"]
        rag_text = exec_context.inputs.context["rag_retrieval"]

        # Return empty if rag_text is None or empty (transient field cleared)
        if not rag_text:
            return ""

        return section_heading + rag_text
    
    def _build_retry_objective_section(self, exec_context: 'ExecutionContext') -> str:
        """Build retry objective context section."""
        section_heading = "\n".join([
            "\n- An agent found the last attempt (see most recent cell in the execution history) to not sufficiently address the active task.",
            "  Consider the following feedback and generate code for another attempt at addressing this task.",
            "  This code will replace the last executed cell. The feedback was:\n\n"
        ])

        # This retrieval is optional - return empty string if not executed
        if "retry_objective" not in exec_context.inputs.context:
            return ""

        # RAG retrieval tool returns a string directly via results["content"]
        rag_text = exec_context.inputs.context["retry_objective"]
        if rag_text is None:
            return ""
        return section_heading + rag_text
    
    def _build_conversation_history_section(self, exec_context: 'ExecutionContext') -> str:
        """Build conversation history section."""
        section_heading = "\n=== Session context: conversation history\n"
        
        conversation_history = exec_context.inputs.context['conversation_history']
        if not conversation_history:
            return ""
        
        # Show last 50 messages to keep context manageable
        recent_conversation = conversation_history[-50:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_conversation])
        
        return section_heading + history_text
    
    def _build_error_section(self, exec_context: 'ExecutionContext') -> str:
        """Build error message section."""
        section_heading = "\n=== Session context: error in last cell\n"

        last_execution_failed = exec_context.inputs.context["last_execution_failed"]
        if last_execution_failed:
            error_message = exec_context.inputs.context["error_message"]
        else:
            error_message = "Last execution passed successfully - without error."
        return section_heading + error_message

    def _build_execution_monitor_section(self, exec_context: 'ExecutionContext') -> str:
        """Build execution monitoring information section."""
        parts = []

        # Cell code
        current_cell = exec_context.inputs.context.get("current_cell", "")
        if current_cell:
            parts.append("=== Cell code currently executing:")
            parts.append("```python")
            parts.append(current_cell)
            parts.append("```")
            parts.append("")

        # Execution information
        elapsed_time = exec_context.inputs.context.get("elapsed_time", 0)
        active_task = exec_context.inputs.context.get("active_task", "Unknown")
        parts.append("=== Execution information:")
        parts.append(f"- Elapsed time: {elapsed_time} seconds")
        parts.append(f"- Active task: {active_task}")
        parts.append("")

        # Partial outputs
        partial_outputs = exec_context.inputs.context.get("partial_outputs", "")
        if partial_outputs:
            parts.append("=== Partial outputs so far:")
            parts.append(partial_outputs)
        else:
            parts.append("=== Partial outputs so far:")
            parts.append("No outputs yet")

        return "\n".join(parts)

    def _build_execution_history_section(self, exec_context: 'ExecutionContext') -> str:
        """Build execution history section with image data removed and output limited."""
        section_heading = "\n".join([
            "\n=== Session context: execution history",
            "This section is divived by cells prefixed with '>', ordered by their time of execution, starting with the latest.",
            "The section of each cell is subdivided into content and outputs, each prefixed with '>>'.",
            "The output section of each cell is further subdivided by output items, each prefixed with '>>>'.\n"
        ])

        execution_history = exec_context.inputs.context['execution_history']
        if execution_history:
            # Show last 10 executions universally to keep prompts manageable
            # Handle both string format (from VSCode) and dict format (from tests)
            formatted_history = []
            for item in execution_history[:10]:
                if isinstance(item, str):
                    formatted_history.append(item)
                elif isinstance(item, dict):
                    # Format dict to string
                    cell_idx = item.get('cell_index', '?')
                    code = item.get('code', '')
                    output = item.get('output', '')
                    formatted_history.append(f"> Cell {cell_idx}\n>> Code:\n{code}\n>> Output:\n{output}")
                else:
                    formatted_history.append(str(item))

            history_text = "Starting from most recent:\n\n" + "\n\n".join(formatted_history)
        else:
            history_text = "No cells have been executed yet."
        return section_heading + history_text

    def _build_notebook_structure_section(self, exec_context: 'ExecutionContext', scenario=None) -> str:
        """Build notebook structure section from VSCode context format."""
        section_heading = "\n".join([
            "\n=== Session context: notebook structure",
            "This section is divived by cells prefixed with '>', order by their appearance in the notebook (the 'index').",
            "The section of each cell is subdivided into content and outputs, each prefixed with '>>'.",
            "The output section of each cell is further subdivided by output items, each prefixed with '>>>'.\n"
        ])

        notebook_structure = exec_context.inputs.context['notebook_structure']
        
        total_cells = notebook_structure['totalCells']
        current_cell_index = exec_context.inputs.context.get('current_cell_index')
        all_cells = notebook_structure['allCells']
        
        if total_cells == 0:
            return "Empty notebook"
        
        # Summary section
        notebook_description = [f"# Total cells: {total_cells}"]
        if current_cell_index is not None:
            notebook_description.append(f"# Currently selected cell: {current_cell_index}")
        notebook_description.extend(all_cells)

        return section_heading + "\n".join(notebook_description)
    
    def _get_json_format_instruction(self, scenario: PromptScenario) -> Optional[str]:
        """Get JSON format instruction from schema for given scenario."""
        from kai.core.orchestration.schemas import SCHEMA_REGISTRY
        
        # Map scenarios to schema registry keys
        scenario_to_schema = {
            PromptScenario.AUTOLOOP_INTENT_CLASSIFICATION: "autoloop_intent_classification",
            PromptScenario.AUTONOMOUS_MARK_COMPLETION: "autonomous_mark_completion",
            PromptScenario.AUTONOMOUS_UPDATE_TASKS: "autonomous_update_tasks",
            PromptScenario.AUTONOMOUS_UPDATE_CRITIQUE: "autonomous_update_critique",
            PromptScenario.BACKTRACK_RECOVERY: "backtrack_recovery",
            PromptScenario.CELL_SELECTION_ADDITION: "cell_positioning",
            PromptScenario.CELL_SELECTION_REPLACEMENT: "cell_positioning",
            PromptScenario.CELL_SELECTION_DELETION_FOR_BACKTRACKING: "cell_selection_deletion",
            PromptScenario.ERROR_RECOVERY: "error_recovery",
            PromptScenario.EXECUTION_MONITOR: "execution_monitor",
            PromptScenario.INTENT_CLASSIFICATION: "intent_classification",
            PromptScenario.REASONING_CRITIQUE: "reasoning_critique",
            PromptScenario.REFERENCE_WORKFLOW_SELECTION: "reference_workflow_selection",
            PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY: "reference_workflow_selection_only",
            PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION: "reference_workflow_cell_selection",
            PromptScenario.SECTION_CODE_REVIEW: "section_code_review",
            PromptScenario.TASK_LIST_GENERATION: "task_list_generation",
            PromptScenario.TASK_LIST_CRITIQUE: "task_list_critique",
        }
        
        schema_key = scenario_to_schema[scenario] if scenario in scenario_to_schema else None
        if schema_key and schema_key in SCHEMA_REGISTRY:
            schema_class = SCHEMA_REGISTRY[schema_key]
            if hasattr(schema_class, 'get_json_format_instruction'):
                return schema_class.get_json_format_instruction()
        
        return None

    def get_system_prompt(self, model_name: str, reasoning_level: Optional[str] = None) -> str:
        """Get system prompt, optionally appending reasoning level directive.

        Args:
            model_name: The LLM model name
            reasoning_level: Reasoning level for OSS models (low/medium/high)

        Returns:
            System prompt with optional reasoning directive appended
        """
        system_prompt = self.SHARED_SYSTEM_PROMPT

        # Append reasoning directive if provided
        if reasoning_level:
            system_prompt = f"{system_prompt}\n\nReasoning: {reasoning_level}"

        return system_prompt
    
    def _model_has_web_search(self, model_name: str) -> bool:
        """Check if model has web search capabilities."""
        web_search_models = ["gpt-oss"]
        return any(model in model_name.lower() for model in web_search_models)
    

