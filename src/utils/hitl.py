"""
hitl.py — Human-in-the-Loop Checkpoint Logic

Implements the required HITL checkpoint where the user can approve
or correct the CNN classification before downstream agents proceed.

Team: Benmouma Salma, Gassi Oumaima
"""

from typing import Optional
from src.utils.logger import AgentLogger


class HumanInTheLoop:
    """
    Human-in-the-Loop checkpoint manager.
    
    Pauses execution to present classification results to the user
    and waits for approval or correction. This satisfies the project
    requirement: "At least 1 checkpoint requiring human approval."
    """

    def __init__(self, logger: Optional[AgentLogger] = None):
        """
        Initialize the HITL manager.

        Args:
            logger: Optional AgentLogger instance for logging HITL actions.
        """
        self.logger = logger or AgentLogger()

    def classification_checkpoint(
        self,
        predicted_class: str,
        confidence: float,
        top_3: list[dict],
        file_path: str
    ) -> dict:
        """
        Present classification results and ask for human approval.

        The user can:
            1. Approve the classification (press Enter or type 'yes')
            2. Override with a different class (type the correct class name)
            3. Reject the document entirely (type 'reject')

        Args:
            predicted_class: The CNN's predicted document class.
            confidence: Confidence score (0-1) of the prediction.
            top_3: Top 3 predictions with class names and scores.
            file_path: Path to the classified document.

        Returns:
            Dictionary with:
                - approved (bool): Whether the classification was approved
                - final_class (str): The final class (original or overridden)
                - was_overridden (bool): Whether the user changed the class
                - rejected (bool): Whether the document was rejected
        """
        # Display classification results
        print("\n" + "=" * 60)
        print("🧑 HUMAN-IN-THE-LOOP CHECKPOINT")
        print("=" * 60)
        print(f"\n📄 Document: {file_path}")
        print(f"\n🔍 Classification Result:")
        print(f"   Predicted Class : {predicted_class}")
        print(f"   Confidence      : {confidence:.1%}")
        print(f"\n   Top 3 Predictions:")
        for i, pred in enumerate(top_3, 1):
            marker = "→" if pred["class"] == predicted_class else " "
            print(f"   {marker} {i}. {pred['class']} ({pred['confidence']:.1%})")

        print(f"\n{'─' * 60}")
        print("Options:")
        print("  [Enter/yes]     → Approve this classification")
        print("  [class name]    → Override with correct class")
        print("  [reject]        → Reject this document")
        print(f"{'─' * 60}")

        # Get user input
        user_input = input("\nYour decision: ").strip().lower()

        # Process decision
        result = {
            "approved": False,
            "final_class": predicted_class,
            "was_overridden": False,
            "rejected": False,
            "original_class": predicted_class,
            "original_confidence": confidence
        }

        if user_input in ("", "yes", "y", "approve"):
            result["approved"] = True
            print(f"\n✅ Classification APPROVED: {predicted_class}")
        elif user_input in ("reject", "r", "no", "n"):
            result["rejected"] = True
            print("\n❌ Document REJECTED by user.")
        else:
            # User is providing a corrected class name
            result["approved"] = True
            result["was_overridden"] = True
            result["final_class"] = user_input
            print(f"\n🔄 Classification OVERRIDDEN: {predicted_class} → {user_input}")

        print("=" * 60 + "\n")

        # Log the HITL decision
        self.logger.log_action(
            agent="HITL_Checkpoint",
            action="classification_review",
            input_data={
                "predicted_class": predicted_class,
                "confidence": confidence,
                "file": file_path
            },
            output_data=result,
            status="success",
            metadata={"user_input": user_input}
        )

        return result

    def generic_checkpoint(self, message: str, data: dict = None) -> bool:
        """
        Generic approval checkpoint for any pipeline step.

        Args:
            message: Message to display to the user.
            data: Optional data to display.

        Returns:
            True if approved, False otherwise.
        """
        print(f"\n{'─' * 60}")
        print(f"🧑 CHECKPOINT: {message}")
        if data:
            for key, value in data.items():
                print(f"   {key}: {value}")
        print(f"{'─' * 60}")

        user_input = input("Approve? [yes/no]: ").strip().lower()
        approved = user_input in ("", "yes", "y")

        self.logger.log_action(
            agent="HITL_Checkpoint",
            action="generic_approval",
            input_data={"message": message, "data": data},
            output_data={"approved": approved},
            status="success"
        )

        return approved
