# Original Multi-Agent Workflow Patterns Section
## Backed up from CLAUDE.md (lines 7-194)

This document preserves the original verbose multi-agent workflow guidance for reference.

---

## Claude Code Preferences

- **Planning model (🏗️ Architect Chief / 🔍 Architect Reviewer)**: Always use **Opus** (Claude Opus 4.6) for planning mode and Plan agents. When using `EnterPlanMode` or `Task` tool with `subagent_type="Plan"`, set `model="opus"` and use `description="🏗️ Architect Chief: [task]"` for initial planning or `description="🔍 Architect Reviewer: [task]"` for plan review and validation to ensure the most thorough architectural analysis and design decisions.

- **Fast model (⚡ Runner)**: Use **Haiku** (Claude Haiku 4.5) for quick, straightforward tasks when using the `Task` tool by setting `model="haiku"` and `description="⚡ Runner: [task]"`. Ideal for:
  - Simple file operations (reading, searching, extracting information)
  - Straightforward edits with clear requirements
  - Repetitive batch operations
  - Quick grep/glob searches
  - Running tests or builds with clear instructions

  **Avoid Haiku for**: Complex debugging, architectural decisions, multi-step problem-solving, or tasks requiring exploration and deeper reasoning.

## Multi-Agent Workflow Patterns

**IMPORTANT**: Use a **separation of concerns** approach where one agent develops code and another agent independently tests or reviews it. This improves code quality and catches issues early.

### Agent Roles and Names

To make multi-agent workflows transparent, **always identify agents by role name**:

- **🏗️ Architect Chief** - Creates initial plans and architectural designs (Opus model, Plan mode)
- **🔍 Architect Reviewer** - Analyzes, validates, and refines architectural plans (Opus model, Plan mode)
- **👨‍💻 Developer** - Main implementation agent (Sonnet model)
- **🧪 Tester** - Runs experiments and verifies functionality (Sonnet/Haiku)
- **👀 Reviewer** - Code review and quality checks (Sonnet model)
- **🔎 Explorer** - Codebase exploration and research (Explore agent)
- **⚡ Runner** - Quick tasks and simple operations (Haiku model)

**Implementation**: Use the agent role name in the `description` parameter of Task tool calls, and have agents identify themselves in their reports.

### Planning → Review Workflow

**For complex architectural tasks, use a two-stage planning workflow** where the Architect Chief creates an initial plan and the Architect Reviewer analyzes and refines it:

1. **Planning Phase** (Architect Chief):
   - Explore the codebase to understand current architecture
   - Design the implementation approach
   - Create a detailed plan with step-by-step instructions
   - Document key decisions and trade-offs
   - Output plan to specs/ folder or present to Architect Reviewer

2. **Review Phase** (Architect Reviewer via Task tool):
   - Analyze the Chief's plan for completeness and correctness
   - Identify potential issues, edge cases, or missing considerations
   - Suggest improvements or alternative approaches
   - Validate that the plan follows project conventions
   - Either approve the plan or provide specific modifications

**Architect Reviewer implementation pattern:**
```python
Task(
    subagent_type="Plan",
    model="opus",
    description="🔍 Architect Reviewer: validate plan",
    prompt="""[AGENT ROLE: 🔍 Architect Reviewer]

Review the architectural plan created by the Architect Chief:

    1. Read the plan document: [path to plan file or describe plan]
    2. Verify completeness: Are all necessary steps covered?
    3. Check correctness: Are the proposed solutions technically sound?
    4. Identify risks: What edge cases or failure modes are missing?
    5. Validate conventions: Does the plan follow project patterns?
    6. Suggest improvements: What could be done better or differently?

    Plan to review: [describe or provide path]
    Focus areas: [list specific concerns to validate]

    Report findings with format: "🔍 Architect Reviewer: [APPROVED/NEEDS REVISION] - [analysis and recommendations]"

    If NEEDS REVISION, provide specific modifications to the plan.
    """
)
```

**When to use two-stage planning:**
- **Complex architectural changes**: Major refactoring, new framework features, performance optimizations
- **High-risk changes**: Modifications to core framework files, dataset preprocessing pipeline changes
- **Multi-file changes**: Tasks affecting 5+ files or multiple subsystems
- **Novel features**: Implementing patterns not already present in the codebase

**Skip for:**
- Simple feature additions following existing patterns
- Bug fixes with clear solutions
- Documentation or spec file updates
- Single-file modifications with straightforward approach

### Development → Testing Workflow

**After implementing any significant change, automatically spawn a testing agent** using the Task tool:

1. **Development Phase** (Main agent):
   - Implement the requested feature following project conventions
   - Complete all code changes
   - **DO NOT run tests yourself** - delegate to testing agent

2. **Testing Phase** (Dedicated testing agent via Task tool):
   - Run relevant experiments to verify functionality
   - Check for runtime errors and edge cases
   - Report results back (success/failure with details)

**Automatically trigger testing agent after:**
- Adding or modifying a GNN layer (`src/models/layers/`)
- Changes to core framework files (`src/framework/`)
- Dataset preprocessing modifications (`src/datasets/`)
- Changes to model configuration logic
- Any feature the user tags as needing verification

**Testing agent implementation pattern:**
```python
Task(
    subagent_type="Bash",  # Use Bash agent for test execution
    model="sonnet",        # Sonnet for reliability, Haiku for simple tests
    description="🧪 Tester: verify implementation",
    prompt="""[AGENT ROLE: 🧪 Tester]

Test the changes by running the appropriate experiment:

    1. Navigate to src directory
    2. Run the relevant example (e.g., python -m examples.basic_example.main)
    3. Check for errors in output
    4. Verify expected results are produced
    5. Report results with format: "🧪 Tester: PASS/FAIL - [details]"

    Experiment to run: [specify based on changes made]
    Expected behavior: [specify what success looks like]
    """
)
```

### Development → Review Workflow

**For complex changes, spawn a code review agent** to analyze the implementation:

```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="👀 Reviewer: analyze code",
    prompt="""[AGENT ROLE: 👀 Reviewer]

Review the recent changes for:

    1. Adherence to project conventions (see Coding Conventions section)
    2. Consistency with existing patterns
    3. Potential bugs or edge cases
    4. Performance considerations
    5. Documentation completeness

    Files to review: [list changed files]
    Focus areas: [specify concerns]

    Report findings with format: "👀 Reviewer: [summary and recommendations]"
    """
)
```

### When to Use Multi-Agent Patterns

**Always use for:**
- New GNN layer implementations (plan → review → develop → test on MUTAG)
- Framework core changes (plan → review → develop → test with multiple experiments)
- Dataset additions (plan → develop → verify data loading and splits)
- Major refactoring or architectural changes (plan → review → develop → test)

**Optional but recommended for:**
- Configuration system changes (plan → develop → test)
- Utility function modifications (develop → test)
- Documentation updates (one agent writes, another reviews)

**Skip for:**
- Trivial changes (typo fixes, comment updates)
- Pure documentation or spec file updates
- Quick debugging iterations
- Simple bug fixes with obvious solutions

### Agent Communication Protocol

- **🏗️ Architect Chief**: After completing plan, state "🏗️ Architect Chief: Plan complete. Spawning 🔍 Architect Reviewer..." Then present plan as "🏗️ Architect Chief: [design decisions and approach]"
- **🔍 Architect Reviewer**: Report as "🔍 Architect Reviewer: ✓ APPROVED - [validation summary]" or "🔍 Architect Reviewer: ⚠ NEEDS REVISION - [specific changes required]"
- **👨‍💻 Developer**: After completing implementation, state "👨‍💻 Developer: Implementation complete. Spawning 🧪 Tester..."
- **🧪 Tester**: Report results as "🧪 Tester: ✓ PASS - [details]" or "🧪 Tester: ✗ FAIL - [error details]"
- **👀 Reviewer**: Report as "👀 Reviewer: [findings and recommendations]"
- **👨‍💻 Developer**: After receiving test/review results, fix issues and re-spawn agents, or confirm success to user

**Important**: Always prefix reports with agent role emoji and name for clarity.
