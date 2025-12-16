"""Simple test for task list preservation bug."""

def test_task_preservation_logic():
    """Test the core logic: preserving all tasks when LLM returns partial list."""
    # Simulate the OLD BUGGY logic
    def buggy_merge(original_tasks, updated_tasks):
        new_tasks = []
        completed_task_ids = set()
        for task in original_tasks:
            if task["status"] != "completed":  # BUG: breaks at first non-completed!
                break
            new_tasks.append(task)
            completed_task_ids.add(task["id"])
        for task in updated_tasks:
            if task["id"] not in completed_task_ids:
                new_tasks.append(task)
        return new_tasks

    # Simulate the FIXED logic
    def fixed_merge(original_tasks, updated_tasks):
        updated_task_map = {task["id"]: task for task in updated_tasks}
        new_tasks = []
        for task in original_tasks:
            task_id = task["id"]
            if task_id in updated_task_map:
                new_tasks.append(updated_task_map[task_id])
            else:
                new_tasks.append(task)
        original_task_ids = {task["id"] for task in original_tasks}
        for task in updated_tasks:
            if task["id"] not in original_task_ids:
                new_tasks.append(task)
        return new_tasks

    # Test case: breastcancer scenario
    original = [
        {"id": 1, "status": "completed"},
        {"id": 2, "status": "completed"},
        {"id": 3, "status": "completed"},
        {"id": 4, "status": "completed"},
        {"id": 5, "status": "pending"},
        {"id": 6, "status": "pending"},
        {"id": 7, "status": "pending"},
    ]

    # LLM returns only completed tasks
    updated = [
        {"id": 1, "status": "completed"},
        {"id": 2, "status": "completed"},
        {"id": 3, "status": "completed"},
        {"id": 4, "status": "completed"},
    ]

    # Buggy result loses pending tasks
    buggy_result = buggy_merge(original, updated)
    assert len(buggy_result) == 4, "Buggy logic kept all 7 tasks (shouldn't happen!)"

    # Fixed result preserves all tasks
    fixed_result = fixed_merge(original, updated)
    assert len(fixed_result) == 7, f"Fixed logic only kept {len(fixed_result)}/7 tasks!"

    # Verify pending tasks preserved
    assert fixed_result[4]["id"] == 5
    assert fixed_result[5]["id"] == 6
    assert fixed_result[6]["id"] == 7

    print("✅ Test passed: Task preservation logic works correctly!")


if __name__ == "__main__":
    test_task_preservation_logic()
