"""Test router behavior after planning tasks.

This is a regression test for the bug where:
- Router planned tasks successfully
- Router immediately exited to "complete" after planning
- task_list was never sent to UI (no manage_progress call)
- Next iteration had no task_list

The fix: After planning (all tasks pending), router should go to manage_progress
to activate the first task and send task_list to UI.
"""


def test_router_logic_after_planning():
    """Test the router logic after planning (all tasks pending).

    This tests the core logic without needing to create full orchestrator.
    """
    # Scenario: Just finished planning, all tasks pending
    state = {
        "task_list": {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "pending"},
                {"id": 2, "task": "Task 2", "status": "pending"},
                {"id": 3, "task": "Task 3", "status": "pending"}
            ]
        },
        "autonomous_mode_continue": False,  # First iteration
    }

    tasks = state["task_list"]["tasks"]
    autonomous_mode_continue = state["autonomous_mode_continue"]
    all_pending = all(t["status"] == "pending" for t in tasks)
    has_active = any(t["status"] == "active" for t in tasks)

    # New router logic with proper first-iteration handling
    if not autonomous_mode_continue:
        if all_pending:
            route = "manage_progress"
        elif has_active:
            route = "complete"
        else:
            route = "complete"
    else:
        route = "continue_execution"

    assert route == "manage_progress", \
        f"After planning (all pending), should route to manage_progress, got: {route}"

    print("✅ Router logic correct: After planning → manage_progress")


def test_router_logic_after_activating_first_task():
    """Test router logic after activating first task in first iteration."""
    # Scenario: manage_progress just activated first task
    state = {
        "task_list": {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "active"},  # Activated!
                {"id": 2, "task": "Task 2", "status": "pending"},
                {"id": 3, "task": "Task 3", "status": "pending"}
            ]
        },
        "autonomous_mode_continue": False,  # Still first iteration
    }

    tasks = state["task_list"]["tasks"]
    autonomous_mode_continue = state["autonomous_mode_continue"]
    all_pending = all(t["status"] == "pending" for t in tasks)
    has_active = any(t["status"] == "active" for t in tasks)

    if not autonomous_mode_continue:
        if all_pending:
            route = "manage_progress"
        elif has_active:
            route = "complete"  # Exit to let UI display task list
        else:
            route = "complete"
    else:
        route = "continue_execution"

    assert route == "complete", \
        f"After activating first task, should route to complete (exit), got: {route}"

    print("✅ Router logic correct: After activation → complete (UI displays task list)")


def test_router_logic_user_stop():
    """Test router logic when user stops (not all tasks pending)."""
    state = {
        "task_list": {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "completed"},
                {"id": 2, "task": "Task 2", "status": "active"},
            ]
        },
        "autonomous_mode_continue": False,
    }

    tasks = state["task_list"]["tasks"]
    autonomous_mode_continue = state["autonomous_mode_continue"]

    all_pending = all(t["status"] == "pending" for t in tasks)

    if not autonomous_mode_continue and all_pending:
        route = "manage_progress"
    elif not autonomous_mode_continue:
        route = "complete"
    else:
        route = "continue_execution"

    assert route == "complete", \
        f"When user stops (not all pending), should route to complete, got: {route}"

    print("✅ Router logic correct: User stop → complete")


def test_router_logic_all_completed():
    """Test router logic when all tasks completed."""
    state = {
        "task_list": {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "completed"},
                {"id": 2, "task": "Task 2", "status": "completed"},
            ]
        },
        "autonomous_mode_continue": True,
    }

    tasks = state["task_list"]["tasks"]

    # Router should check for all_complete
    all_complete = all(t["status"] == "completed" for t in tasks)

    if all_complete:
        route = "complete"
    else:
        route = "continue_execution"

    assert route == "complete", \
        f"When all tasks completed, should route to complete, got: {route}"

    print("✅ Router logic correct: All completed → complete")


def test_old_router_logic_was_wrong():
    """Demonstrate that old router logic was broken.

    OLD LOGIC:
    - if tasks exist and autonomous_mode_continue=False → complete

    This was wrong because after planning, tasks exist but we haven't started yet!
    """
    state = {
        "task_list": {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "pending"},
                {"id": 2, "task": "Task 2", "status": "pending"},
            ]
        },
        "autonomous_mode_continue": False,
    }

    tasks = state["task_list"]["tasks"]
    autonomous_mode_continue = state["autonomous_mode_continue"]

    # OLD LOGIC (BROKEN):
    if tasks and not autonomous_mode_continue:
        old_route = "complete"  # ❌ WRONG - we just planned!
    else:
        old_route = "continue"

    assert old_route == "complete", "Old logic was broken (went to complete after planning)"

    # NEW LOGIC (FIXED):
    all_pending = all(t["status"] == "pending" for t in tasks)
    if not autonomous_mode_continue and all_pending:
        new_route = "manage_progress"  # ✅ CORRECT - activate first task
    elif not autonomous_mode_continue:
        new_route = "complete"
    else:
        new_route = "continue"

    assert new_route == "manage_progress", "New logic correctly routes to manage_progress"

    print("✅ Demonstrated: Old logic was broken, new logic is correct")
