package main

import "fmt"

type taskOption struct {
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Type        string   `json:"type"`
	Choices     []string `json:"choices,omitempty"`
	Default     any      `json:"default,omitempty"`
	Required    bool     `json:"required,omitempty"`
	Description string   `json:"description,omitempty"`
}

type taskInfo struct {
	ID      string       `json:"id"`
	Name    string       `json:"name"`
	Options []taskOption `json:"options"`
}

type tasksResponse struct {
	Tasks []taskInfo `json:"tasks"`
}

func init() {
	register(command{
		group: "task", name: "list",
		summary: "List all server tasks with their option schemas (GET /tasks)",
		run: func(a *App, args []string) int {
			var out tasksResponse
			if err := a.Client.DoJSON("GET", "/tasks", nil, &out); err != nil {
				return a.Fail(err)
			}
			if a.Table {
				return a.PrintJSON(out.Tasks)
			}
			return a.PrintJSON(out)
		},
	})
	register(command{
		group: "task", name: "show", args: "<id>",
		summary: "Show one task's option schema",
		run: func(a *App, args []string) int {
			if len(args) != 1 {
				return a.Usage(nil, "usage: lokictl task show <id>")
			}
			var out tasksResponse
			if err := a.Client.DoJSON("GET", "/tasks", nil, &out); err != nil {
				return a.Fail(err)
			}
			for _, t := range out.Tasks {
				if t.ID == args[0] {
					return a.PrintJSON(t)
				}
			}
			return a.Fail(fmt.Errorf("unknown task %q — see: lokictl task list", args[0]))
		},
	})
}
