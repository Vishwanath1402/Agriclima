# AgriClima — Climate-Resilient Farming Advisor
# Powered by Gemma 4
#
# To run locally or on Hugging Face Spaces:
#   pip install -r requirements.txt
#   python app.py
#
# To deploy on Hugging Face Spaces:
#   1. Create a new Space (gradio template)
#   2. Upload this app.py and requirements.txt
#   3. The Space will build and host it automatically

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- CONFIG -----
# Use the Hugging Face Hub model ID. Change to the exact ID Google publishes
# for the Gemma 4 E4B variant. Common pattern: "google/gemma-4-e4b-it"
MODEL_ID = "google/gemma-4-e4b-it"

SYSTEM_PROMPT = """You are AgriClima, a climate-resilient farming advisor for smallholder farmers.

Your job: give practical, specific, actionable advice that helps farmers adapt to changing climate conditions.

Rules:
- Use simple language. Avoid technical jargon.
- Give concrete recommendations, not vague principles.
- If the farmer's situation has serious risk (e.g., crop failure imminent), say so clearly first.
- Suggest 2-3 alternative crops or practices when relevant, naming specific varieties or species suited to the farmer's region when you can.
- Always end with: a single "What to do this week" action line.
- If you don't know something specific to their region, say so honestly.

When the input is too vague to give region-specific advice (e.g., crop is unspecified, no concrete weather details, or location is too broad), do not guess. Instead:
- Acknowledge what's missing in 1-2 sentences
- Ask 2-3 specific follow-up questions that would let you give better advice
- Then provide one piece of safe general guidance the farmer can act on while gathering that info
- Skip the standard three-section format in this case

Otherwise, when the input is specific enough, format your answer in three short sections:
1. **Risk assessment** (1-2 sentences)
2. **Recommendations** (2-4 bullet points)
3. **What to do this week** (one line)
"""

# ----- MODEL LOADING (happens once at startup) -----
print("Loading Gemma 4...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.")


# ----- INFERENCE FUNCTION -----
def advise(location, crop, observation, question):
    if not all([location, crop, observation, question]):
        return "Please fill in all four fields so the advisor can give useful guidance."

    user_msg = f"""A farmer needs advice.

Location/region: {location}
Current or planned crop: {crop}
Recent weather/climate observation: {observation}
Specific question or concern: {question}

Give your advice now."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = out[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----- GRADIO UI -----
with gr.Blocks(title="AgriClima — Climate-Resilient Farming Advisor") as demo:
    gr.Markdown(
        """
        # 🌱 AgriClima
        ### A climate-resilient farming advisor for smallholder farmers
        Powered by Gemma 4 — designed to work in low-connectivity settings.
        """
    )

    with gr.Row():
        with gr.Column():
            location = gr.Textbox(
                label="Your region",
                placeholder="e.g., Punjab, India  •  Western Kenya  •  Northern Ghana",
            )
            crop = gr.Textbox(
                label="Crop you grow / plan to grow",
                placeholder="e.g., wheat, maize, sorghum, cassava",
            )
            observation = gr.Textbox(
                label="What you've noticed about the weather/climate recently",
                placeholder="e.g., Rains came one month later than normal. Days are hotter.",
                lines=3,
            )
            question = gr.Textbox(
                label="Your specific question or concern",
                placeholder="e.g., Should I still plant wheat this season?",
                lines=2,
            )
            submit_btn = gr.Button("Get advice", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="AgriClima says:")

    submit_btn.click(
        fn=advise,
        inputs=[location, crop, observation, question],
        outputs=output,
    )

    gr.Examples(
        examples=[
            ["Punjab, India", "wheat",
             "Winter is 4°C warmer than the 10-year average. Soil is unusually dry.",
             "Should I still plant wheat next season or switch?"],
            ["Western Kenya", "maize",
             "The long rains have been arriving 3 weeks later than usual for two years. Some neighbors lost their crop last season.",
             "When should I plant this season, and should I be doing anything different?"],
            ["Northern Ghana", "cassava (4 months into growth)",
             "An unusual heatwave has temperatures 8 degrees above normal for 10 days. Leaves are wilting visibly.",
             "Is my crop in immediate danger and what can I do this week to save it?"],
            ["central India", "vegetables",
             "weather has been unpredictable",
             "what should I grow next season?"],
        ],
        inputs=[location, crop, observation, question],
        label="Try one of these example questions",
    )

    gr.Markdown(
        "*AgriClima provides general guidance only. Always confirm major decisions "
        "with local agricultural extension officers when possible.*"
    )

if __name__ == "__main__":
    demo.launch()
