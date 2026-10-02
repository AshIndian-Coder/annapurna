package services

import (
	"fmt"
)

type NotFoundError struct {
	Resource string
	ID       string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("%s not found: %s", e.Resource, e.ID)
}

type ConflictError struct {
	Code    string
	Message string
}

func (e *ConflictError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

type ForbiddenError struct {
	Message string
}

func (e *ForbiddenError) Error() string {
	return e.Message
}
