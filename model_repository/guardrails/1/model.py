# # model.py — Triton Python Backend for NeMo Guardrails
# # -----------------------------------------------------

# import numpy as np
# import triton_python_backend_utils as pb_utils
# from nemoguardrails.rails import LLMRails
# from nemoguardrails.rails.llm.config import RailsConfig


# class TritonPythonModel:

#     def initialize(self, args):
#         """
#         Called once when Triton loads the model.
#         """
#         model_dir = args["model_repository"]
#         version = args["model_version"]

#         rails_path = f"{model_dir}/{version}/rails"

#         # Load Guardrails configuration (config.yml + rails.co)
#         self.config = RailsConfig.from_path(rails_path)

#         # Initialize NeMo Guardrails engine (v0.16)
#         self.rails = LLMRails(config=self.config)

#         print("NeMo Guardrails 0.16 loaded successfully inside Triton.")

#     def execute(self, requests):
#         """
#         Handles incoming inference requests.
#         """
#         responses = []

#         for req in requests:
#             # Read input tensor
#             text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
#             raw_text = text_tensor.as_numpy()[0].decode("utf-8")

#             # Call NeMo Guardrails (v0.16 API) 
#             # → returns {"output": "..."}
#             guardrails_result = self.rails.generate(raw_text)

#             bot_reply = guardrails_result.get(
#                 "output", "Error: No guardrails output"
#             )

#             # Convert to Triton output
#             out_tensor = pb_utils.Tensor(
#                 "OUTPUT_TEXT",
#                 np.array([bot_reply.encode("utf-8")], dtype=object)
#             )

#             responses.append(
#                 pb_utils.InferenceResponse(output_tensors=[out_tensor])
#             )

#         return responses

#     def finalize(self):
#         """Called when Triton unloads the model."""
#         print(" Triton Guardrails model shutting down.")


# model.py — Triton Python Backend for NeMo Guardrails
# -----------------------------------------------------

# import numpy as np
# import triton_python_backend_utils as pb_utils

# from nemoguardrails.rails import LLMRails
# from nemoguardrails.rails.llm.config import RailsConfig


# class TritonPythonModel:
#     def initialize(self, args):
#         """
#         Called once when Triton loads the model.
#         """
#         model_dir = args["model_repository"]
#         version = args["model_version"]

#         rails_path = f"{model_dir}/{version}/rails"

#         # Load rails config correctly
#         self.config = RailsConfig.from_path(rails_path)

#         # Initialize NeMo Guardrails
#         self.rails = LLMRails(config=self.config)

#         print("NeMo Guardrails loaded successfully inside Triton.")

#     def execute(self, requests):
#         responses = []

#         for req in requests:
#             text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
#             raw_text = text_tensor.as_numpy()[0].decode("utf-8")

#             # THIS is where Guardrails calls your LLM (Ollama)
#             result = self.rails.generate(
#                 messages=[{"role": "user", "content": raw_text}]
#             )

#             bot_reply = result.get("output_text", "")

#             out_tensor = pb_utils.Tensor(
#                 "OUTPUT_TEXT",
#                 np.array([bot_reply.encode("utf-8")], dtype=object)
#             )

#             responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

#         return responses

#     def finalize(self):
#         print("Triton Guardrails model shutting down.")


import numpy as np
import triton_python_backend_utils as pb_utils

from nemoguardrails.rails import LLMRails
from nemoguardrails.rails.llm.config import RailsConfig


class TritonPythonModel:
    def initialize(self, args):
        """
        Called once when Triton loads the model.
        """
        print("===== TRITON PYTHON BACKEND INITIALIZE START =====")

        model_dir = args.get("model_repository", "")
        version = args.get("model_version", "")
        print(f"[DEBUG] model_repository: {model_dir}")
        print(f"[DEBUG] model_version: {version}")

        rails_path = f"{model_dir}/{version}/rails"
        print(f"[DEBUG] Loading NeMo Rails from path: {rails_path}")

        # Load rails config
        self.config = RailsConfig.from_path(rails_path)
        print("[DEBUG] RailsConfig loaded successfully.")

        # Initialize NeMo Guardrails
        self.rails = LLMRails(config=self.config)
        print("[DEBUG] LLMRails initialized successfully.")

        print("===== TRITON INITIALIZATION COMPLETE =====\n")

    def execute(self, requests):
        print("===== TRITON EXECUTION START =====")
        print(f"[DEBUG] Number of requests received: {len(requests)}")

        responses = []

        for req_id, req in enumerate(requests):
            print(f"\n--- Processing Request #{req_id} ---")

            # READ INPUT
            text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
            if text_tensor is None:
                print("[ERROR] Missing TEXT input tensor!")
                bot_reply = "Error: TEXT input tensor missing."
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("OUTPUT_TEXT", np.array([bot_reply.encode("utf-8")], dtype=object))
                        ]
                    )
                )
                continue

            raw_text = text_tensor.as_numpy()[0].decode("utf-8")
            print(f"[DEBUG] Input TEXT received: {raw_text}")

            # CALL NeMo Guardrails
            print("[DEBUG] Calling NeMo Guardrails generate()...")
            try:
                result = self.rails.generate(
                    messages=[{"role": "user", "content": raw_text}]
                )
                print(f"[DEBUG] Guardrails response raw: {result}")
            except Exception as e:
                print("[ERROR] Guardrails crashed:", e)
                result = {"content": f"Guardrails Error: {str(e)}"}

            # ------------------------------------------
            # FIX: extract the correct content response
            # ------------------------------------------
            bot_reply = result.get("content", "")
            if not bot_reply:
                # fallback in case of rare alt key
                bot_reply = result.get("output_text", "")

            print(f"[DEBUG] Final bot reply after guardrails: {bot_reply}")

            # CREATE OUTPUT TENSOR
            out_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                np.array([bot_reply.encode("utf-8")], dtype=object)
            )
            print("[DEBUG] OUTPUT_TEXT tensor created.")

            # SEND RESPONSE
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            print(f"[DEBUG] Response for request #{req_id} added.")

        print("===== TRITON EXECUTION END =====\n")
        return responses

    def finalize(self):
        print("===== TRITON MODEL FINALIZE CALLED =====")
