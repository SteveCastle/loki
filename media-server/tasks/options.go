package tasks

import (
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/jobqueue"
)

// TaskOption describes a single configurable parameter for a task.
type TaskOption struct {
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Type        string   `json:"type"` // "string", "bool", "enum", "multi-enum", "number"
	Choices     []string `json:"choices,omitempty"`
	Default     any      `json:"default,omitempty"`
	Required    bool     `json:"required,omitempty"`
	Description string   `json:"description,omitempty"`
}

// ParseOptions extracts typed option values from a job's Arguments slice,
// falling back to each option's default when the flag is absent.
func ParseOptions(j *jobqueue.Job, options []TaskOption) map[string]any {
	result := make(map[string]any)
	for _, opt := range options {
		if opt.Default != nil {
			result[opt.Name] = opt.Default
		} else {
			switch opt.Type {
			case "bool":
				result[opt.Name] = false
			case "number":
				result[opt.Name] = 0.0
			default:
				result[opt.Name] = ""
			}
		}
	}

	optMap := make(map[string]*TaskOption, len(options))
	for i := range options {
		optMap[options[i].Name] = &options[i]
	}

	args := j.Arguments
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "--") {
			continue
		}
		key := strings.TrimPrefix(arg, "--")
		value := ""
		hasEquals := false
		if eqIdx := strings.Index(key, "="); eqIdx >= 0 {
			value = key[eqIdx+1:]
			key = key[:eqIdx]
			hasEquals = true
		}
		opt, ok := optMap[key]
		if !ok {
			continue
		}
		switch opt.Type {
		case "bool":
			if hasEquals {
				result[key] = value == "true" || value == "1" || value == "yes"
			} else {
				result[key] = true
			}
		case "number":
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				result[key] = n
			}
		default:
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			result[key] = value
		}
	}
	return result
}
