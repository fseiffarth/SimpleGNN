# Claude Code - Multi-Agent Workflow Patterns

This document defines reusable multi-agent workflow patterns for Claude Code that can be applied to any software project.

## Claude Code Model Selection

- **Planning model (🏗️ Architect Chief / 🔍 Architect Reviewer)**: Always use **Opus** (Claude Opus 4.6) for planning mode and Plan agents. When using `EnterPlanMode` or `Task` tool with `subagent_type="Plan"`, set `model="opus"` and use `description="🏗️ Architect Chief: [task]"` for initial planning or `description="🔍 Architect Reviewer: [task]"` for plan review and validation to ensure the most thorough architectural analysis and design decisions.

- **Standard model (👨‍💻 Developer / 🧪 Tester / 👀 Reviewer)**: Use **Sonnet** (Claude Sonnet 4.5) for implementation, testing, and code review tasks. This provides a good balance of capability and cost.

- **Fast model (⚡ Runner)**: Use **Haiku** (Claude Haiku 4.5) for quick, straightforward tasks when using the `Task` tool by setting `model="haiku"` and `description="⚡ Runner: [task]"`. Ideal for:
  - Simple file operations (reading, searching, extracting information)
  - Straightforward edits with clear requirements
  - Repetitive batch operations
  - Quick grep/glob searches
  - Running tests or builds with clear instructions

  **Avoid Haiku for**: Complex debugging, architectural decisions, multi-step problem-solving, or tasks requiring exploration and deeper reasoning.

## Multi-Agent Workflow Philosophy

**IMPORTANT**: Use a **separation of concerns** approach where one agent develops code and another agent independently tests or reviews it. This improves code quality and catches issues early.

**Benefits:**
- Independent verification reduces bias and catches issues the implementer might miss
- Parallel execution when possible (planning + review, multiple independent test suites)
- Clear accountability and traceability of who did what
- Better context management (each agent focuses on their specific task)

## Agent Roles and Names

To make multi-agent workflows transparent, **always identify agents by role name**:

- **🏗️ Architect Chief** - Creates initial plans and architectural designs (Opus model, Plan mode)
- **🔍 Architect Reviewer** - Analyzes, validates, and refines architectural plans (Opus model, Plan mode)
- **👨‍💻 Developer** - Main implementation agent (Sonnet model)
- **🧪 Tester** - Runs experiments and verifies functionality (Sonnet/Haiku)
- **👀 Reviewer** - Code review and quality checks (Sonnet model)
- **🔎 Explorer** - Codebase exploration and research (Explore agent)
- **⚡ Runner** - Quick tasks and simple operations (Haiku model)

**Implementation**: Use the agent role name in the `description` parameter of Task tool calls, and have agents identify themselves in their reports.

## Core Workflow Patterns

### Planning → Review Workflow

**For complex architectural tasks, use a two-stage planning workflow** where the Architect Chief creates an initial plan and the Architect Reviewer analyzes and refines it:

1. **Planning Phase** (Architect Chief):
   - Explore the codebase to understand current architecture
   - Design the implementation approach
   - Create a detailed plan with step-by-step instructions
   - Document key decisions and trade-offs
   - Output plan to project specs/docs folder or present to Architect Reviewer

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
- **Complex architectural changes**: Major refactoring, new features requiring design decisions, performance optimizations
- **High-risk changes**: Modifications to core systems, breaking API changes, data migration
- **Multi-file changes**: Tasks affecting 5+ files or multiple subsystems
- **Novel features**: Implementing patterns not already present in the codebase
- **Cross-cutting concerns**: Changes affecting multiple layers of the application

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
   - Run relevant tests to verify functionality
   - Check for runtime errors and edge cases
   - Report results back (success/failure with details)

**Testing agent implementation pattern:**
```python
Task(
    subagent_type="Bash",  # Use Bash agent for test execution
    model="sonnet",        # Sonnet for reliability, Haiku for simple tests
    description="🧪 Tester: verify implementation",
    prompt="""[AGENT ROLE: 🧪 Tester]

Test the changes by running the appropriate test suite:

    1. Navigate to the relevant directory
    2. Run the test suite (e.g., pytest, npm test, cargo test)
    3. Check for errors in output
    4. Verify expected results are produced
    5. Report results with format: "🧪 Tester: PASS/FAIL - [details]"

    Tests to run: [specify test files/suites based on changes made]
    Expected behavior: [specify what success looks like]
    """
)
```

**Automatically trigger testing agent after:**
- Adding or modifying core functionality
- Changes to critical business logic
- API endpoint modifications
- Database schema or query changes
- Any feature the user tags as needing verification

### Development → Review Workflow

**For complex changes, spawn a code review agent** to analyze the implementation:

```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="👀 Reviewer: analyze code",
    prompt="""[AGENT ROLE: 👀 Reviewer]

Review the recent changes for:

    1. Adherence to project conventions (coding style, naming, structure)
    2. Consistency with existing patterns
    3. Potential bugs or edge cases
    4. Performance considerations
    5. Security vulnerabilities (SQL injection, XSS, auth issues)
    6. Error handling and validation
    7. Documentation completeness

    Files to review: [list changed files]
    Focus areas: [specify concerns]

    Report findings with format: "👀 Reviewer: [summary and recommendations]"
    """
)
```

## When to Use Multi-Agent Patterns

### Always use for:
- **New feature implementations**: Non-trivial features requiring design decisions (plan → review → develop → test)
- **Core system changes**: Modifications to critical infrastructure (plan → review → develop → test)
- **Data handling changes**: Database migrations, API contracts, data pipelines (plan → develop → test)
- **Major refactoring**: Architectural changes affecting multiple components (plan → review → develop → test)

### Optional but recommended for:
- **Configuration system changes**: (plan → develop → test)
- **Utility function modifications**: (develop → test)
- **Documentation updates**: (one agent writes, another reviews)
- **Non-trivial bug fixes**: Bugs requiring investigation (explore → develop → test)

### Skip for:
- **Trivial changes**: Typo fixes, comment updates, whitespace changes
- **Pure documentation**: README updates, comment additions (unless substantial)
- **Quick debugging iterations**: Printf debugging, temporary logging
- **Simple bug fixes**: Obvious one-line fixes with clear root cause

## Agent Communication Protocol

Agents should identify themselves clearly in all communications:

- **🏗️ Architect Chief**: After completing plan, state "🏗️ Architect Chief: Plan complete. Spawning 🔍 Architect Reviewer..." Then present plan as "🏗️ Architect Chief: [design decisions and approach]"
- **🔍 Architect Reviewer**: Report as "🔍 Architect Reviewer: ✓ APPROVED - [validation summary]" or "🔍 Architect Reviewer: ⚠ NEEDS REVISION - [specific changes required]"
- **👨‍💻 Developer**: After completing implementation, state "👨‍💻 Developer: Implementation complete. Spawning 🧪 Tester..."
- **🧪 Tester**: Report results as "🧪 Tester: ✓ PASS - [details]" or "🧪 Tester: ✗ FAIL - [error details]"
- **👀 Reviewer**: Report as "👀 Reviewer: [findings and recommendations]"
- **🔎 Explorer**: Report as "🔎 Explorer: [findings and relevant file locations]"
- **⚡ Runner**: Report as "⚡ Runner: [task completed with brief result]"
- **👨‍💻 Developer**: After receiving test/review results, fix issues and re-spawn agents, or confirm success to user

**Important**: Always prefix reports with agent role emoji and name for clarity.

## Advanced Patterns

### Parallel Agent Execution

When tasks are independent, spawn multiple agents in parallel:

```python
# Example: Test multiple independent modules simultaneously
Task(subagent_type="Bash", model="haiku", description="🧪 Tester: backend tests", ...)
Task(subagent_type="Bash", model="haiku", description="🧪 Tester: frontend tests", ...)
Task(subagent_type="Bash", model="haiku", description="🧪 Tester: integration tests", ...)
```

### Iterative Refinement

For complex tasks, use multiple rounds of review:

1. **🏗️ Architect Chief** creates initial plan
2. **🔍 Architect Reviewer** reviews and suggests improvements
3. **👨‍💻 Developer** implements based on approved plan
4. **👀 Reviewer** reviews implementation
5. **👨‍💻 Developer** addresses feedback
6. **🧪 Tester** verifies final implementation

### Background Agents

For long-running tasks (large test suites, builds), use background execution:

```python
Task(
    subagent_type="Bash",
    model="haiku",
    description="🧪 Tester: full integration suite",
    run_in_background=True,
    prompt="Run the complete integration test suite and report results"
)
```

## Best Practices

1. **Always spawn testing agents after implementation**: Don't test your own code; maintain separation of concerns
2. **Use appropriate models**: Opus for planning, Sonnet for implementation, Haiku for simple tasks
3. **Be specific in prompts**: Give agents clear success criteria and expected outputs
4. **Chain workflows**: Plan → Review → Develop → Test → Review for critical features
5. **Document agent decisions**: Have agents write to spec/docs folders for future reference
6. **Fail fast**: If a tester reports failures, fix immediately before proceeding
7. **Parallel when possible**: Independent tasks should run concurrently
8. **Clear communication**: Always use role prefixes in agent reports

## Customizing for Your Project

To adapt these patterns to your specific project:

1. **Create a project-specific CLAUDE.md** that includes:
   - Link to this general pattern document
   - Project structure and conventions
   - Tech stack and dependencies
   - When to trigger which workflows (e.g., "Always test after modifying X")
   - Project-specific agent prompts

2. **Define test strategies**: What tests should the 🧪 Tester run for different types of changes?

3. **Establish review criteria**: What should the 👀 Reviewer check in your codebase?

4. **Document conventions**: What patterns should the 🔍 Architect Reviewer validate against?

5. **Create workflow triggers**: When should two-stage planning be used vs. simple develop-test?

## Example: Full Feature Implementation

Here's a complete workflow for adding a new feature:

```
User Request: "Add user authentication to the API"

1. 🏗️ Architect Chief (Opus, Plan mode)
   - Explores codebase
   - Designs auth approach (JWT, session storage, middleware)
   - Creates implementation plan in specs/auth-implementation.md

2. 🔍 Architect Reviewer (Opus, Plan agent)
   - Reviews plan for security issues
   - Validates against project patterns
   - Approves with minor security recommendations

3. 👨‍💻 Developer (Sonnet, main agent)
   - Implements auth middleware
   - Adds login/logout endpoints
   - Updates database schema

4. 🧪 Tester (Sonnet, Bash agent)
   - Runs unit tests for auth module
   - Runs integration tests for protected endpoints
   - Reports: "🧪 Tester: ✓ PASS - All 23 tests passing"

5. 👀 Reviewer (Sonnet, general-purpose agent)
   - Reviews for security vulnerabilities
   - Checks error handling
   - Reports: "👀 Reviewer: Approved with suggestion to add rate limiting"

6. 👨‍💻 Developer (Sonnet, main agent)
   - Adds rate limiting based on review feedback
   - Final verification by spawning tester again
```

---

**Remember**: The goal of multi-agent workflows is to improve code quality through independent verification, not to add unnecessary complexity. Use these patterns judiciously based on the complexity and risk of each task.
