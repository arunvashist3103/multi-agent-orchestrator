from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ResearchRequest(BaseModel):
    """Payload to initiate a new research workflow."""

    model_config = ConfigDict(str_strip_whitespace=True)

    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research topic or question to investigate.",
        examples=["Latest advancements in grid-scale battery storage"],
    )

class ReviewDecision(BaseModel):
    """Human-in-the-loop review decision payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["approve", "edit", "reject"]
    edited_content: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=5000,
        description="Required when type is 'edit'. Replaces the generated bullet points.",
    )
    feedback: Optional[str] = Field(
        default=None,
        min_length=10,
        max_length=1000,
        description="Required when type is 'reject'. Guides the researcher's next pass.",
    )
    @model_validator(mode="after")
    def validate_decision_requirements(self) -> "ReviewDecision":
        if self.type == "edit" and not self.edited_content:
            raise ValueError(
                "edited_content is required (minimum 20 characters) when type is 'edit'."
            )
        if self.type == "reject" and not self.feedback:
            raise ValueError(
                "feedback is required (minimum 10 characters) when type is 'reject'."
            )
        return self

class PendingReview(BaseModel):
    """Response returned when the workflow pauses for human review."""

    thread_id: str
    status: Literal["pending_review"] = "pending_review"
    bullet_points: str
    redo_count_so_far: int
    redos_remaining: int


class FinalAnswer(BaseModel):
    """Response returned once the writer agent completes the final synthesis."""

    thread_id: str
    status: Literal["complete"] = "complete"
    final_answer: str
    
