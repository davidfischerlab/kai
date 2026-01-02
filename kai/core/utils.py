def format_task_list(task_list) -> str:
    if task_list is None or "tasks" not in task_list:
        return ""

    task_text = ""
    for task in task_list["tasks"]:
        task_text += f"Task ID={task['id']}, status={task['status']}: {task['task']}\n"
    return task_text
