# model.py — Triton Python Backend for NeMo Guardrails
# -----------------------------------------------------

import numpy as np
import triton_python_backend_utils as pb_utils
from nemoguardrails.rails import LLMRails
from nemoguardrails.rails.llm.config import RailsConfig


class TritonPythonModel:

    def initialize(self, args):
        """
        Called once when Triton loads the model.
        """
        model_dir = args["model_repository"]
        version = args["model_version"]

        rails_path = f"{model_dir}/{version}/rails"

        # Load Guardrails configuration (config.yml + rails.co)
        self.config = RailsConfig.from_path(rails_path)

        # Initialize NeMo Guardrails engine (v0.16)
        self.rails = LLMRails(config=self.config)

        print("NeMo Guardrails 0.16 loaded successfully inside Triton.")

    def execute(self, requests):
        """
        Handles incoming inference requests.
        """
        responses = []

        for req in requests:
            # Read input tensor
            text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
            raw_text = text_tensor.as_numpy()[0].decode("utf-8")

            # Call NeMo Guardrails (v0.16 API) 
            # → returns {"output": "..."}
            guardrails_result = self.rails.generate(raw_text)

            bot_reply = guardrails_result.get(
                "output", "Error: No guardrails output"
            )

            # Convert to Triton output
            out_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                np.array([bot_reply.encode("utf-8")], dtype=object)
            )

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses

    def finalize(self):
        """Called when Triton unloads the model."""
        print(" Triton Guardrails model shutting down.")
