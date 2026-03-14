"""
VibeCoder Models Module

This module contains the data models and schemas used throughout the VibeCoder application.
It defines Pydantic models for request/response validation, database models, and data
transfer objects (DTOs) that facilitate communication between different layers of the
application.

The models in this module ensure data integrity, provide automatic validation,
and enable seamless serialization/deserialization of data structures used in
the VibeCoder autonomous coding agent system.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    has_tests: bool = False
    requirements: Optional[List[str]] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    language: str
    framework: Optional[str]
    has_tests: bool
    status: ProjectStatus
    created_at: str
    updated_at: str

class TaskCreate(BaseModel):
    mission: str
    description: str
    step: str
    target_file: Optional[str] = None
    project_id: str

class TaskResponse(BaseModel):
    id: str
    mission: str
    description: str
    step: str
    target_file: Optional[str]
    status: TaskStatus
    project_id: str
    created_at: str
    updated_at: str
    result: Optional[str] = None

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: str
    framework: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class CodeGenerationResponse(BaseModel):
    code: str
    explanation: Optional[str] = None
    suggestions: Optional[List[str]] = None
